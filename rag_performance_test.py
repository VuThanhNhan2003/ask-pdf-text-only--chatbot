import os
import json
import time
import glob
from datetime import datetime
from typing import List, Dict, Any
import traceback
from processor import RAGProcessor

class RAGPerformanceTester:
    """Test suite for evaluating RAG pipeline performance"""
    
    def __init__(self, data_folder: str = "./data"):
        self.data_folder = data_folder
        self.results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_folder": data_folder
            },
            "document_processing": {},
            "chunk_analysis": {},
            "query_performance": {},
            "overall_metrics": {}
        }
        
    def find_pdf_files(self) -> List[str]:
        """Find all PDF files in the data folder"""
        pdf_pattern = os.path.join(self.data_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {self.data_folder}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")
        
        return pdf_files
    
    def test_document_processing(self, pdf_files: List[str]) -> RAGProcessor:
        """Test document processing performance"""
        print("\n=== Testing Document Processing ===")
        
        start_time = time.time()
        
        try:
            # Initialize RAG processor
            processor = RAGProcessor(pdf_files)
            
            processing_time = time.time() - start_time
            
            # Get document info
            doc_info = processor.get_document_info()
            
            self.results["document_processing"] = {
                "success": True,
                "processing_time_seconds": round(processing_time, 2),
                "total_documents": len(pdf_files),
                "document_details": doc_info,
                "total_pages": sum(doc["pages"] for doc in doc_info)
            }
            
            print(f"✓ Document processing completed in {processing_time:.2f} seconds")
            print(f"✓ Processed {len(doc_info)} documents with {sum(doc['pages'] for doc in doc_info)} total pages")
            
            return processor
            
        except Exception as e:
            self.results["document_processing"] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time_seconds": time.time() - start_time
            }
            print(f"✗ Document processing failed: {e}")
            raise
    
    def extract_and_analyze_chunks(self, processor: RAGProcessor) -> None:
        """Extract chunks from the vector store for analysis"""
        print("\n=== Analyzing Chunks ===")
        
        try:
            # Get the vector store from the processor
            vector_store = processor.conversation_chain.retriever.vectorstore
            
            # Get all texts from the vector store
            # Note: This is a simplified approach - in practice, you might need to access internal structures
            all_texts = []
            if hasattr(vector_store, 'index_to_docstore_id'):
                for i in range(len(vector_store.index_to_docstore_id)):
                    doc_id = vector_store.index_to_docstore_id[i]
                    if doc_id in vector_store.docstore._dict:
                        doc = vector_store.docstore._dict[doc_id]
                        all_texts.append({
                            "chunk_id": i,
                            "text": doc.page_content,
                            "length": len(doc.page_content),
                            "metadata": getattr(doc, 'metadata', {})
                        })
            
            # Analyze chunks
            chunk_lengths = [chunk["length"] for chunk in all_texts]
            
            self.results["chunk_analysis"] = {
                "total_chunks": len(all_texts),
                "average_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                "chunks_detail": all_texts
            }
            
            print(f"✓ Analyzed {len(all_texts)} chunks")
            print(f"✓ Average chunk length: {self.results['chunk_analysis']['average_chunk_length']} characters")
            
        except Exception as e:
            self.results["chunk_analysis"] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"✗ Chunk analysis failed: {e}")
    
    def test_query_performance(self, processor: RAGProcessor, test_questions: List[str]) -> None:
        """Test query response performance"""
        print("\n=== Testing Query Performance ===")
        
        query_results = []
        total_start_time = time.time()
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nTesting Question {i}/{len(test_questions)}: {question[:50]}...")
            
            start_time = time.time()
            
            try:
                # Get response
                response = processor.get_response(question)
                response_time = time.time() - start_time
                
                query_result = {
                    "question_id": i,
                    "question": question,
                    "response": response,
                    "response_time_seconds": round(response_time, 2),
                    "response_length": len(response),
                    "success": True
                }
                
                print(f"  ✓ Answered in {response_time:.2f} seconds")
                print(f"  ✓ Response length: {len(response)} characters")
                print(f"  → Response: {response}")  # 👈 thêm dòng này

                
            except Exception as e:
                query_result = {
                    "question_id": i,
                    "question": question,
                    "error": str(e),
                    "response_time_seconds": round(time.time() - start_time, 2),
                    "success": False
                }
                print(f"  ✗ Failed in {time.time() - start_time:.2f} seconds: {e}")
            
            query_results.append(query_result)
        
        total_query_time = time.time() - total_start_time
        
        # Calculate metrics
        successful_queries = [q for q in query_results if q["success"]]
        failed_queries = [q for q in query_results if not q["success"]]
        
        if successful_queries:
            avg_response_time = sum(q["response_time_seconds"] for q in successful_queries) / len(successful_queries)
            avg_response_length = sum(q["response_length"] for q in successful_queries) / len(successful_queries)
        else:
            avg_response_time = 0
            avg_response_length = 0
        
        self.results["query_performance"] = {
            "total_questions": len(test_questions),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": round(len(successful_queries) / len(test_questions) * 100, 2),
            "total_query_time_seconds": round(total_query_time, 2),
            "average_response_time_seconds": round(avg_response_time, 2),
            "average_response_length": round(avg_response_length, 2),
            "detailed_results": query_results
        }
        
        print(f"\n✓ Query testing completed")
        print(f"✓ Success rate: {self.results['query_performance']['success_rate']}%")
        print(f"✓ Average response time: {avg_response_time:.2f} seconds")
    
    def calculate_overall_metrics(self) -> None:
        """Calculate overall performance metrics"""
        print("\n=== Calculating Overall Metrics ===")
        
        # Extract key metrics
        doc_processing_time = self.results["document_processing"].get("processing_time_seconds", 0)
        total_pages = self.results["document_processing"].get("total_pages", 0)
        total_chunks = self.results["chunk_analysis"].get("total_chunks", 0)
        avg_response_time = self.results["query_performance"].get("average_response_time_seconds", 0)
        success_rate = self.results["query_performance"].get("success_rate", 0)
        
        self.results["overall_metrics"] = {
            "pages_per_second": round(total_pages / doc_processing_time, 2) if doc_processing_time > 0 else 0,
            "chunks_per_page": round(total_chunks / total_pages, 2) if total_pages > 0 else 0,
            "processing_efficiency_score": round((success_rate / 100) * (1 / max(avg_response_time, 0.1)), 2),
            "memory_usage_estimate": f"{total_chunks * 1.5:.1f} KB",  # Rough estimate
            "recommended_improvements": self._generate_recommendations()
        }
        
        print(f"✓ Processing speed: {self.results['overall_metrics']['pages_per_second']} pages/second")
        print(f"✓ Chunking ratio: {self.results['overall_metrics']['chunks_per_page']} chunks/page")
        print(f"✓ Efficiency score: {self.results['overall_metrics']['processing_efficiency_score']}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        avg_response_time = self.results["query_performance"].get("average_response_time_seconds", 0)
        success_rate = self.results["query_performance"].get("success_rate", 100)
        avg_chunk_length = self.results["chunk_analysis"].get("average_chunk_length", 0)
        
        if avg_response_time > 10:
            recommendations.append("Consider reducing chunk size or retrieval count (k) for faster responses")
        
        if success_rate < 90:
            recommendations.append("Review failed queries and consider improving chunk overlap or prompt template")
        
        if avg_chunk_length > 2000:
            recommendations.append("Consider reducing chunk size for better granularity")
        elif avg_chunk_length < 500:
            recommendations.append("Consider increasing chunk size for better context")
        
        if not recommendations:
            recommendations.append("Performance looks good! Consider testing with more diverse queries")
        
        return recommendations
    
    def export_results(self, output_file: str = "rag_performance_results.json") -> None:
        """Export test results to JSON file"""
        print(f"\n=== Exporting Results to {output_file} ===")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results exported to {output_file}")
            
            # Also create a summary file
            summary_file = output_file.replace('.json', '_summary.txt')
            self._create_summary_report(summary_file)
            
        except Exception as e:
            print(f"✗ Failed to export results: {e}")
    
    def _create_summary_report(self, summary_file: str) -> None:
        """Create a human-readable summary report"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("RAG PERFORMANCE TEST SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                
                # Document Processing
                f.write("DOCUMENT PROCESSING:\n")
                doc_proc = self.results["document_processing"]
                f.write(f"- Processing Time: {doc_proc.get('processing_time_seconds', 0)} seconds\n")
                f.write(f"- Total Documents: {doc_proc.get('total_documents', 0)}\n")
                f.write(f"- Total Pages: {doc_proc.get('total_pages', 0)}\n\n")
                
                # Chunk Analysis
                f.write("CHUNK ANALYSIS:\n")
                chunk_analysis = self.results["chunk_analysis"]
                f.write(f"- Total Chunks: {chunk_analysis.get('total_chunks', 0)}\n")
                f.write(f"- Average Chunk Length: {chunk_analysis.get('average_chunk_length', 0)} chars\n")
                f.write(f"- Min/Max Chunk Length: {chunk_analysis.get('min_chunk_length', 0)}/{chunk_analysis.get('max_chunk_length', 0)} chars\n\n")
                
                # Query Performance
                f.write("QUERY PERFORMANCE:\n")
                query_perf = self.results["query_performance"]
                f.write(f"- Success Rate: {query_perf.get('success_rate', 0)}%\n")
                f.write(f"- Average Response Time: {query_perf.get('average_response_time_seconds', 0)} seconds\n")
                f.write(f"- Average Response Length: {query_perf.get('average_response_length', 0)} chars\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                recommendations = self.results["overall_metrics"].get("recommended_improvements", [])
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            print(f"✓ Summary report created: {summary_file}")
            
        except Exception as e:
            print(f"✗ Failed to create summary report: {e}")
    
    def run_full_test(self, test_questions: List[str] = None) -> None:
        """Run the complete test suite"""
        print("Starting RAG Performance Test Suite")
        print("=" * 50)
        
        # Default test questions if none provided
        if test_questions is None:
            test_questions = [
            "Phát biểu luận điểm Maxwell thứ nhất và giải thích tại sao điện trường xoáy lại có đường sức khép kín?",
            "Viết phương trình Maxwell-Faraday dưới dạng tích phân và giải thích ý nghĩa vật lý của từng thành phần trong phương trình.",
            "Tại sao Maxwell lại đưa ra khái niệm 'dòng điện dịch'? Phân tích thí nghiệm với tụ điện và nguồn xoay chiều để chứng minh sự tồn tại của dòng điện dịch.",
            "Tính vận tốc truyền sóng điện từ trong chân không biết ε₀ = 8.85×10⁻¹² F/m và μ₀ = 4π×10⁻⁷ H/m.",
            "So sánh dòng điện dẫn và dòng điện dịch về bản chất, tính chất và vai trò trong việc tạo ra từ trường.",
            "Từ phương trình Maxwell dưới dạng tích phân, hãy suy ra phương trình Maxwell-Ampère dưới dạng vi phân: rot H⃗ = j⃗dẫn + ∂D⃗/∂t",
            "Chứng minh rằng sóng điện từ phẳng là sóng ngang và các vectơ E⃗, B⃗, n⃗ lập thành tam diện thuận.",
            "Viết biểu thức vectơ Poynting và giải thích ý nghĩa vật lý của nó trong quá trình truyền năng lượng sóng điện từ."
            ]
        
        try:
            # Find PDF files
            pdf_files = self.find_pdf_files()
            if not pdf_files:
                print("No PDF files found. Please add PDF files to the data folder.")
                return
            
            # Test document processing
            processor = self.test_document_processing(pdf_files)
            
            # Analyze chunks
            self.extract_and_analyze_chunks(processor)
            
            # Test query performance
            self.test_query_performance(processor, test_questions)
            
            # Calculate overall metrics
            self.calculate_overall_metrics()
            
            # Export results
            self.export_results()
            
            print("\n" + "=" * 50)
            print("RAG Performance Test Suite Completed Successfully!")
            print("Check the generated JSON and summary files for detailed results.")
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            print("Check the error details in the exported results file.")
            self.export_results("rag_performance_results_error.json")


def main():
    """Main function to run the performance test"""
    
    # Initialize tester
    tester = RAGPerformanceTester(data_folder="./data")  # Change this path if needed
    
    # Define your test questions here
    test_questions = [
        "Phát biểu luận điểm Maxwell thứ nhất và giải thích tại sao điện trường xoáy lại có đường sức khép kín?",
        "Viết phương trình Maxwell-Faraday dưới dạng tích phân và giải thích ý nghĩa vật lý của từng thành phần trong phương trình.",
        "Tại sao Maxwell lại đưa ra khái niệm 'dòng điện dịch'? Phân tích thí nghiệm với tụ điện và nguồn xoay chiều để chứng minh sự tồn tại của dòng điện dịch.",
        "Tính vận tốc truyền sóng điện từ trong chân không biết ε₀ = 8.85×10⁻¹² F/m và μ₀ = 4π×10⁻⁷ H/m.",
        "So sánh dòng điện dẫn và dòng điện dịch về bản chất, tính chất và vai trò trong việc tạo ra từ trường.",
        "Từ phương trình Maxwell dưới dạng tích phân, hãy suy ra phương trình Maxwell-Ampère dưới dạng vi phân: rot H⃗ = j⃗dẫn + ∂D⃗/∂t",
        "Chứng minh rằng sóng điện từ phẳng là sóng ngang và các vectơ E⃗, B⃗, n⃗ lập thành tam diện thuận.",
        "Viết biểu thức vectơ Poynting và giải thích ý nghĩa vật lý của nó trong quá trình truyền năng lượng sóng điện từ."
    ]
    
    # Run the full test suite
    tester.run_full_test(test_questions)


if __name__ == "__main__":
    main()