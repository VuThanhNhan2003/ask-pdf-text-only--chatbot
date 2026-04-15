# Whisper API + RAG TextNode Pipeline

Hệ thống gồm hai pipeline độc lập, kết nối thành một luồng xử lý từ video thô đến dữ liệu RAG sẵn sàng index:

1. **ASR pipeline** — video/audio → subtitle `.vtt` + transcript `.txt`
2. **TextNode pipeline** — transcript `.txt` → `.textnodes.json` cho RAG

---

## 1. Tổng quan kiến trúc

| Thành phần | File | Vai trò |
|---|---|---|
| API server | `api.py`, `main.py` | FastAPI, nhận request, quản lý job |
| ASR core | `pipeline.py` | Download, transcribe, filter, refine, output |
| Schemas | `schemas.py` | Request/response + `TranscriptionOptions` |
| TextNode pipeline | `rag_textnode_pipeline.py` | Chuyển transcript thành TextNode JSON |
| ASR output | `file_vtt/<video_name>/` | `.vtt` + `.txt` |
| TextNode input | `file_transcript/` | Thư mục đọc `.txt` (copy từ `file_vtt` hoặc custom) |
| TextNode output | `file_textnodes/` | `<course>.textnodes.json` |

> ASR pipeline và TextNode pipeline tách biệt hoàn toàn. ASR ghi vào `file_vtt/`, TextNode pipeline đọc từ `file_transcript/` (mặc định). Copy/sync transcript thủ công hoặc dùng `--input-root` để chỉ định thư mục khác.

---

## 2. ASR Pipeline — Video → VTT / TXT

### 2.1 Luồng xử lý trong `process_video_transcription`

```
Job queue
  └─ Worker thread picks job
       ├─ Download media
       │    ├─ Direct URL   → requests.get() với retry
       │    ├─ M3U8/HLS     → ffmpeg -i ... -c copy
       │    └─ Google Drive → Drive API + MediaIoBaseDownload
       ├─ Load Faster-Whisper (thread-local model cache)
       ├─ Transcribe với anti-hallucination params + optional VAD
       │    └─ initial_prompt = build_initial_prompt(language, subject)
       ├─ Filter: validate_segment_quality + clean_repetitive_text
       ├─ [Optional] LLM refine — enable_llm_refine=true
       │    ├─ Batch segments
       │    ├─ [Optional] Query RAG context từ k segment trước (CB-RAG)
       │    │    └─ Inject context vào system prompt của Qwen3
       │    └─ Qwen3 sửa chính tả, thuật ngữ, dấu câu
       └─ Generate output
            ├─ *.vtt  — smart_segment_split (50–70 chars/segment)
            └─ *.txt  — clean transcript (có/không timestamp)
```

### 2.2 Input được hỗ trợ

- Direct media URL (`.mp4`, `.avi`, `.mov`, ...)
- M3U8 / HLS stream
- Google Drive file link
- Google Drive folder link (endpoint `/convert-drive-folder`)

### 2.3 Job store và recovery

- Trạng thái job được persist tại `file_vtt/.jobs_status.json`
- Payload job được persist tại `file_vtt/.job_payloads.json`
- Khi server restart: job `queued/processing` được tự động re-queue nếu payload còn; ngược lại bị mark `failed`
- Queue capacity: `JOB_QUEUE_MAXSIZE` (mặc định `max(100, 2000)`)

### 2.4 API Endpoints

#### `GET /health`

Trả về trạng thái server, ffmpeg, queue/worker metrics.

#### `POST /convert`

Queue một job chuyển đổi video.

```json
{
  "video_url": "https://example.com/video.mp4",
  "language": "vi",
  "model": "large-v3-turbo",
  "enable_vad": true,
  "condition_on_previous_text": false,
  "beam_size": 5,
  "temperature": 0.0,
  "compression_ratio_threshold": 2.4,
  "no_speech_threshold": 0.6,
  "enable_llm_refine": true,
  "llm_proxy_url": "http://127.0.0.1:5000",
  "llm_model": "Qwen/Qwen3-8B-AWQ",
  "llm_timeout_seconds": 60,
  "refine_batch_size": 20,
  "prompt_template": null,
  "txt_include_timestamps": false,
  "subject": "Môn Triết học Mác-Lênin",
  "rag_api_url": "http://127.0.0.1:9100",
  "rag_context_window": 5
}
```

Khi `rag_api_url` được cung cấp, mỗi batch refinement sẽ tự động query RAG với transcript của `rag_context_window` segment trước để lấy context thuật ngữ từ slide bài giảng đã index — không cần cung cấp keyword thủ công.

#### `POST /convert-drive-folder`

Queue toàn bộ video trong một Google Drive folder.

```json
{
  "folder_url": "https://drive.google.com/drive/folders/<FOLDER_ID>",
  "language": "vi",
  "enable_llm_refine": true,
  "subject": "Môn Triết học Mác-Lênin",
  "rag_api_url": "http://127.0.0.1:9100"
}
```

#### `GET /status/{job_id}`

Trả về trạng thái và result payload của job.

#### `GET /list`

Liệt kê tất cả file `.vtt` / `.txt` trong `file_vtt/`.

#### `GET /download/{filename}`

Download file output.

#### `DELETE /jobs/{job_id}`

Xóa job khỏi memory store.

---

## 3. TextNode Pipeline — TXT → TextNodes cho RAG

Script: `rag_textnode_pipeline.py`

### 3.1 Mục tiêu

Chuyển transcript `.txt` thành TextNode JSON chất lượng cao cho RAG:

- Granularity nguyên tử: mỗi node = đúng 1 ý chính
- Loại bỏ filler, meta-discourse, ASR noise
- Keyword, topic, QA template hợp lệ
- Deduplication mạnh (Jaccard + char-ngram + topic similarity)

### 3.2 Luồng xử lý trong `run_pipeline`

```
Scan *.txt under --input-root
  └─ Group files theo course folder (level-1 subfolder)
       └─ Per file:
            ├─ split_transcript_for_llm (900–1400 words, 70-word overlap)
            ├─ call_llm_for_textnodes → raw TextNode JSON
            ├─ sanitize_node:
            │    ├─ clean_node_text (bỏ filler, meta, lặp)
            │    ├─ should_drop_node (length, repetition, orphan, meta-discourse)
            │    ├─ quality_gate_node (min words, lexical overlap, unique ratio)
            │    ├─ validate_keywords (bigram-aware, loại ASR noise)
            │    ├─ validate_question_templates
            │    └─ [optional] provenance: source_coverage + source_quote
            ├─ dedupe_nodes (Jaccard + char-4gram + topic sim)
            ├─ merge_short_nodes_by_topic
            ├─ dedupe_nodes (lần 2)
            └─ [Fallback] nếu collapse về ≤1 node với transcript ≥500 words:
                 smaller chunks + force_multi_nodes=True
  └─ Cross-file dedupe per course
  └─ Write output-root/<course>.textnodes.json
```

### 3.3 Chạy script

```bash
# Cơ bản
python3 rag_textnode_pipeline.py \
  --input-root file_transcript \
  --output-root file_textnodes \
  --llm-proxy-url http://127.0.0.1:5000 \
  --llm-model Qwen/Qwen3-8B-AWQ

# Khuyến nghị — quality-oriented
python3 rag_textnode_pipeline.py \
  --input-root file_transcript \
  --output-root file_textnodes \
  --llm-proxy-url http://127.0.0.1:5000 \
  --llm-model Qwen/Qwen3-8B-AWQ \
  --include-provenance \
  --quality-min-words 28 \
  --quality-min-overlap 0.52 \
  --quality-min-unique-ratio 0.30 \
  --dedupe-threshold 0.76 \
  --file-max-retries 2
```

### 3.4 Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--input-chunk-min-words` | 900 | Số từ tối thiểu mỗi chunk gửi LLM |
| `--input-chunk-max-words` | 1400 | Số từ tối đa mỗi chunk |
| `--input-chunk-overlap-words` | 70 | Overlap giữa các chunk liên tiếp |
| `--quality-min-words` | 28 | Node phải có tối thiểu n từ |
| `--quality-min-overlap` | 0.52 | Lexical overlap tối thiểu node↔source |
| `--quality-min-unique-ratio` | 0.30 | Tỉ lệ unique token tối thiểu |
| `--dedupe-threshold` | 0.76 | Jaccard threshold để loại near-duplicate |
| `--file-max-retries` | 2 | Số lần retry toàn bộ file khi LLM lỗi |
| `--include-provenance` | false | Thêm `source_coverage` + `source_quote` vào metadata |

### 3.5 Cấu trúc thư mục input

```
file_transcript/
└─ Môn Triết học Mác-Lênin/
    ├─ Buoi_1.txt
    ├─ Buoi_2.txt
    └─ ...
```

Tên thư mục level-1 (`Môn Triết học Mác-Lênin`) được dùng làm `subject` trong metadata và là tên file output.

### 3.6 Output format

```json
[
  {
    "id": "uuid",
    "text": "Nội dung học thuật đã chuẩn hóa từ văn nói sang văn viết...",
    "metadata": {
      "subject": "Môn Triết học Mác-Lênin",
      "page": null,
      "topic": "Tên chủ đề ngắn gọn",
      "category": "Theory",
      "keywords": ["thuật ngữ 1", "thuật ngữ 2"],
      "has_code": false,
      "file_name": "Buoi_1.txt",
      "question_templates": [
        "X là gì?",
        "Tại sao X quan trọng?",
        "X khác Y ở điểm nào?"
      ],
      "source_coverage": 0.63,
      "source_quote": "..."
    }
  }
]
```

`source_coverage` và `source_quote` chỉ có khi dùng `--include-provenance`.

`category` chỉ nhận một trong ba giá trị: `Theory`, `Example`, `Process`.

---

## 4. Luồng vận hành khuyến nghị

```
1. POST /convert hoặc /convert-drive-folder
      └─ Sinh *.vtt + *.txt vào file_vtt/<video_name>/

2. Copy/sync *.txt sang file_transcript/<course>/

3. python3 rag_textnode_pipeline.py ...
      └─ Sinh file_textnodes/<course>.textnodes.json

4. Index textnodes vào RAG system (Qdrant)
      └─ Sẵn sàng phục vụ query
```

---

## 5. Chạy bằng Docker

```bash
# Build và start
docker compose up -d --build

# Xem logs
docker compose logs -f whisper-api

# Health check
curl http://127.0.0.1:8000/health

# Stop
docker compose down
```

---

## 6. Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `LLM_PROXY_URL` | `http://host.docker.internal:5000` | URL LLM proxy cho ASR refine |
| `RAG_API_URL` | _(rỗng)_ | URL RAG API để query context tự động |
| `TRANSCRIPTION_WORKERS` | `auto` | Số worker thread xử lý job |
| `WHISPER_NUM_WORKERS` | `2` | Số worker nội bộ của Whisper |
| `JOB_QUEUE_MAXSIZE` | `2000` | Dung lượng tối đa job queue |
| `DOWNLOAD_MAX_RETRIES` | `4` | Số lần retry download |
| `DOWNLOAD_RETRY_DELAY_SECONDS` | `2` | Thời gian chờ giữa các lần retry |
| `DIRECT_DOWNLOAD_CONNECT_TIMEOUT` | `20` | Timeout kết nối (giây) |
| `DIRECT_DOWNLOAD_READ_TIMEOUT` | `300` | Timeout đọc dữ liệu (giây) |
| `GOOGLE_API_RETRIES` | `5` | Số lần retry Google Drive API |
| `GOOGLE_DRIVE_API_KEY` | _(rỗng)_ | API key Drive (thay thế cho service account) |

---

## 7. Google Drive credentials

Đặt `credentials.json` ở thư mục gốc project. Hỗ trợ hai chế độ:

- **Service account JSON** (khuyến nghị) — cần share folder/file Drive cho email service account
- **JSON với field `api_key`/`key`** — cho tài nguyên public

Hoặc dùng biến môi trường `GOOGLE_DRIVE_API_KEY`.
