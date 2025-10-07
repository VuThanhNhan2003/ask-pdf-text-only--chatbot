"""
Authentication and user management service
"""
import bcrypt
import secrets
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from database.models import User, UserSettings


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


class AuthService:
    """Handle user authentication and management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    
    @staticmethod
    def create_user(
        db: Session,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """
        Create new user account
        
        Raises:
            AuthenticationError: If email or username already exists
        """
        # Check if email exists
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            raise AuthenticationError("Email already registered")
        
        # Check if username exists
        existing_username = db.query(User).filter(User.username == username).first()
        if existing_username:
            raise AuthenticationError("Username already taken")
        
        # Validate password
        if len(password) < 6:
            raise AuthenticationError("Password must be at least 6 characters")
        
        # Create user
        user = User(
            email=email.lower(),
            username=username,
            password_hash=AuthService.hash_password(password),
            full_name=full_name
        )
        
        db.add(user)
        db.flush()  # Get user.id
        
        # Create default settings
        settings = UserSettings(user_id=user.id)
        db.add(settings)
        
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def authenticate_user(
        db: Session,
        email: str,
        password: str
    ) -> User:
        """
        Authenticate user with email and password
        
        Returns:
            User object if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        user = db.query(User).filter(User.email == email.lower()).first()
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        if not AuthService.verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email.lower()).first()
    
    @staticmethod
    def update_password(
        db: Session,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Update user password
        
        Raises:
            AuthenticationError: If old password is incorrect
        """
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise AuthenticationError("User not found")
        
        if not AuthService.verify_password(old_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")
        
        if len(new_password) < 6:
            raise AuthenticationError("New password must be at least 6 characters")
        
        user.password_hash = AuthService.hash_password(new_password)
        db.commit()
        
        return True
    
    @staticmethod
    def generate_reset_token() -> str:
        """Generate secure reset token"""
        return secrets.token_urlsafe(32)