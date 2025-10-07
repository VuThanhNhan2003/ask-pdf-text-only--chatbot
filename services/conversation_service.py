"""
Conversation and message management service
"""
import json
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from database.models import Conversation, Message, SharedConversation, User


class ConversationService:
    """Handle conversation operations"""
    
    @staticmethod
    def create_conversation(
        db: Session,
        user_id: int,
        subject: Optional[str] = None,
        title: str = "New Chat"
    ) -> Conversation:
        """Create new conversation"""
        conversation = Conversation(
            user_id=user_id,
            title=title,
            subject=subject
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    
    @staticmethod
    def get_conversation(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> Optional[Conversation]:
        """Get conversation by ID (with user verification)"""
        return db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.is_archived == False
        ).first()
    
    @staticmethod
    def get_user_conversations(
        db: Session,
        user_id: int,
        include_archived: bool = False,
        limit: int = 50
    ) -> List[Conversation]:
        """Get all conversations for a user"""
        query = db.query(Conversation).filter(Conversation.user_id == user_id)
        
        if not include_archived:
            query = query.filter(Conversation.is_archived == False)
        
        return query.order_by(
            desc(Conversation.is_pinned),
            desc(Conversation.updated_at)
        ).limit(limit).all()
    
    @staticmethod
    def update_conversation_title(
        db: Session,
        conversation_id: int,
        user_id: int,
        title: str
    ) -> bool:
        """Update conversation title"""
        conversation = ConversationService.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return False
        
        conversation.title = title
        conversation.updated_at = datetime.utcnow()
        db.commit()
        return True
    
    @staticmethod
    def auto_generate_title(
        db: Session,
        conversation_id: int,
        first_message: str
    ) -> str:
        """Auto-generate conversation title from first message"""
        # Take first 50 chars and clean up
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."
        
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if conversation:
            conversation.title = title
            db.commit()
        
        return title
    
    @staticmethod
    def toggle_pin(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> bool:
        """Toggle conversation pin status"""
        conversation = ConversationService.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return False
        
        conversation.is_pinned = not conversation.is_pinned
        db.commit()
        return True
    
    @staticmethod
    def archive_conversation(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> bool:
        """Archive a conversation"""
        conversation = ConversationService.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return False
        
        conversation.is_archived = True
        db.commit()
        return True
    
    @staticmethod
    def delete_conversation(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> bool:
        """Permanently delete a conversation"""
        conversation = ConversationService.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return False
        
        db.delete(conversation)
        db.commit()
        return True
    
    @staticmethod
    def search_conversations(
        db: Session,
        user_id: int,
        query: str,
        limit: int = 20
    ) -> List[Conversation]:
        """Search conversations by title or content"""
        search_term = f"%{query}%"
        
        # Search in conversation titles and message content
        conversations = db.query(Conversation).join(Message).filter(
            Conversation.user_id == user_id,
            Conversation.is_archived == False,
            or_(
                Conversation.title.ilike(search_term),
                Message.content.ilike(search_term)
            )
        ).distinct().order_by(
            desc(Conversation.updated_at)
        ).limit(limit).all()
        
        return conversations


class MessageService:
    """Handle message operations"""
    
    @staticmethod
    def add_message(
        db: Session,
        conversation_id: int,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        tokens_used: Optional[int] = None,
        processing_time: Optional[int] = None
    ) -> Message:
        """Add message to conversation"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=json.dumps(sources) if sources else None,
            tokens_used=tokens_used,
            processing_time=processing_time
        )
        
        db.add(message)
        
        # Update conversation timestamp
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if conversation:
            conversation.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        
        return message
    
    @staticmethod
    def get_conversation_messages(
        db: Session,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get all messages in a conversation"""
        query = db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).order_by(Message.created_at)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def delete_message(
        db: Session,
        message_id: int
    ) -> bool:
        """Soft delete a message"""
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            return False
        
        message.is_deleted = True
        db.commit()
        return True
    
    @staticmethod
    def get_message_count(db: Session, conversation_id: int) -> int:
        """Get message count for a conversation"""
        return db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).count()


class ShareService:
    """Handle conversation sharing"""
    
    @staticmethod
    def create_share_link(
        db: Session,
        conversation_id: int,
        user_id: int,
        expires_days: Optional[int] = 30
    ) -> Optional[str]:
        """Create shareable link for conversation"""
        # Verify ownership
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        
        if not conversation:
            return None
        
        # Check if already shared
        existing = db.query(SharedConversation).filter(
            SharedConversation.conversation_id == conversation_id
        ).first()
        
        if existing and existing.is_active:
            return existing.share_token
        
        # Create new share
        share_token = secrets.token_urlsafe(16)
        expires_at = datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        
        shared = SharedConversation(
            conversation_id=conversation_id,
            share_token=share_token,
            expires_at=expires_at
        )
        
        db.add(shared)
        db.commit()
        
        return share_token
    
    @staticmethod
    def get_shared_conversation(
        db: Session,
        share_token: str
    ) -> Optional[Conversation]:
        """Get conversation by share token"""
        shared = db.query(SharedConversation).filter(
            SharedConversation.share_token == share_token,
            SharedConversation.is_active == True
        ).first()
        
        if not shared:
            return None
        
        # Check expiration
        if shared.expires_at and shared.expires_at < datetime.utcnow():
            return None
        
        # Increment view count
        shared.view_count += 1
        db.commit()
        
        return db.query(Conversation).filter(
            Conversation.id == shared.conversation_id
        ).first()
    
    @staticmethod
    def revoke_share(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> bool:
        """Revoke share link"""
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        
        if not conversation:
            return False
        
        shared = db.query(SharedConversation).filter(
            SharedConversation.conversation_id == conversation_id
        ).first()
        
        if shared:
            shared.is_active = False
            db.commit()
        
        return True