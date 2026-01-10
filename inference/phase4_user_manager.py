"""
🆔 PHASE 4.1: USER MANAGEMENT SYSTEM
====================================

Purpose: Manual user selection and management
Author: Psychologist AI System
Date: January 10, 2026

Features:
    - Manual user selection
    - User registration
    - User profiles management
    - Separate data files per user
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


class UserProfile:
    """Represents a registered user"""
    
    def __init__(self, user_id: str, name: str, created_at: str, 
                 last_active: str = None, total_sessions: int = 0):
        self.user_id = user_id
        self.name = name
        self.created_at = created_at
        self.last_active = last_active or created_at
        self.total_sessions = total_sessions
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'total_sessions': self.total_sessions
        }
    
    @staticmethod
    def from_dict(data):
        return UserProfile(**data)


class UserManager:
    """
    Manages multiple users in the system
    
    Features:
        - Register new users
        - List existing users
        - Select active user
        - Delete users (with confirmation)
    """
    
    def __init__(self, storage_dir: str = "data/user_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.storage_dir / "users.json"
        self.users: Dict[str, UserProfile] = {}
        
        self._load_users()
    
    def _load_users(self):
        """Load registered users from file"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                data = json.load(f)
                self.users = {
                    uid: UserProfile.from_dict(profile)
                    for uid, profile in data.items()
                }
    
    def _save_users(self):
        """Save registered users to file"""
        data = {
            uid: profile.to_dict()
            for uid, profile in self.users.items()
        }
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_user(self, name: str) -> str:
        """
        Register a new user
        
        Args:
            name: User's display name
            
        Returns:
            user_id: Generated unique ID
        """
        # Generate user ID from name + timestamp
        timestamp = datetime.now().isoformat()
        user_id = self._generate_user_id(name, timestamp)
        
        # Create profile
        profile = UserProfile(
            user_id=user_id,
            name=name,
            created_at=timestamp,
            total_sessions=0
        )
        
        self.users[user_id] = profile
        self._save_users()
        
        print(f"✓ Registered user: {name} (ID: {user_id})")
        return user_id
    
    def _generate_user_id(self, name: str, timestamp: str) -> str:
        """Generate unique user ID"""
        # Create hash from name + timestamp
        hash_input = f"{name}_{timestamp}".encode()
        hash_value = hashlib.sha256(hash_input).hexdigest()[:12]
        
        # Sanitize name for ID
        safe_name = "".join(c if c.isalnum() else "_" for c in name.lower())
        
        return f"{safe_name}_{hash_value}"
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.users.get(user_id)
    
    def list_users(self) -> List[UserProfile]:
        """Get all registered users"""
        return sorted(self.users.values(), key=lambda u: u.last_active, reverse=True)
    
    def update_last_active(self, user_id: str):
        """Update user's last active timestamp"""
        if user_id in self.users:
            self.users[user_id].last_active = datetime.now().isoformat()
            self._save_users()
    
    def increment_session_count(self, user_id: str):
        """Increment user's total session count"""
        if user_id in self.users:
            self.users[user_id].total_sessions += 1
            self._save_users()
    
    def delete_user(self, user_id: str, delete_data: bool = True):
        """
        Delete user (requires confirmation)
        
        Args:
            user_id: User to delete
            delete_data: If True, also delete memory files
        """
        if user_id not in self.users:
            print(f"✗ User {user_id} not found")
            return
        
        user = self.users[user_id]
        
        # Delete from registry
        del self.users[user_id]
        self._save_users()
        
        # Delete data files if requested
        if delete_data:
            memory_file = self.storage_dir / f"{user_id}_longterm_memory.json"
            if memory_file.exists():
                memory_file.unlink()
        
        print(f"✓ Deleted user: {user.name}")


class UserSelector:
    """
    Interactive user selection interface
    
    Usage:
        selector = UserSelector()
        user_id = selector.select_user()
    """
    
    def __init__(self, storage_dir: str = "data/user_memory"):
        self.manager = UserManager(storage_dir)
    
    def select_user(self) -> str:
        """
        Show interactive user selection
        
        Returns:
            user_id: Selected or newly created user ID
        """
        users = self.manager.list_users()
        
        print("\n" + "="*60)
        print("👤 USER SELECTION")
        print("="*60)
        
        if not users:
            print("No registered users found.")
            return self._register_new_user()
        
        # Display existing users
        print("\nRegistered Users:")
        for i, user in enumerate(users, 1):
            last_active = datetime.fromisoformat(user.last_active)
            time_ago = self._format_time_ago(last_active)
            
            print(f"  {i}. {user.name}")
            print(f"     Last active: {time_ago} | Sessions: {user.total_sessions}")
        
        print(f"\n  {len(users)+1}. Register New User")
        print("  0. Exit")
        
        # Get selection
        while True:
            try:
                choice = input("\nSelect user (number): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print("Exiting...")
                    exit(0)
                elif choice_num == len(users) + 1:
                    return self._register_new_user()
                elif 1 <= choice_num <= len(users):
                    selected_user = users[choice_num - 1]
                    self.manager.update_last_active(selected_user.user_id)
                    print(f"\n✓ Selected: {selected_user.name}")
                    return selected_user.user_id
                else:
                    print("Invalid selection. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid input. Try again.")
    
    def _register_new_user(self) -> str:
        """Register a new user interactively"""
        print("\n" + "-"*60)
        print("📝 NEW USER REGISTRATION")
        print("-"*60)
        
        while True:
            name = input("Enter your name: ").strip()
            
            if not name:
                print("Name cannot be empty. Try again.")
                continue
            
            if len(name) < 2:
                print("Name too short. Try again.")
                continue
            
            # Confirm
            confirm = input(f"Register as '{name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                user_id = self.manager.register_user(name)
                return user_id
            else:
                print("Registration cancelled.")
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format time difference as human-readable string"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            return f"{diff.days // 365} year(s) ago"
        elif diff.days > 30:
            return f"{diff.days // 30} month(s) ago"
        elif diff.days > 0:
            return f"{diff.days} day(s) ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} hour(s) ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60} minute(s) ago"
        else:
            return "just now"


# ============================================================================
# CLI UTILITY
# ============================================================================

def manage_users_cli():
    """Command-line user management utility"""
    manager = UserManager()
    
    while True:
        print("\n" + "="*60)
        print("👥 USER MANAGEMENT")
        print("="*60)
        print("1. List users")
        print("2. Register new user")
        print("3. Delete user")
        print("4. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            users = manager.list_users()
            if not users:
                print("\nNo registered users.")
            else:
                print(f"\nTotal Users: {len(users)}")
                print("-"*60)
                for user in users:
                    print(f"\n{user.name} (ID: {user.user_id})")
                    print(f"  Created: {user.created_at}")
                    print(f"  Last active: {user.last_active}")
                    print(f"  Total sessions: {user.total_sessions}")
        
        elif choice == '2':
            name = input("Enter name: ").strip()
            if name:
                manager.register_user(name)
        
        elif choice == '3':
            users = manager.list_users()
            if not users:
                print("\nNo users to delete.")
                continue
            
            print("\nRegistered Users:")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user.name}")
            
            try:
                idx = int(input("\nUser to delete (number): ")) - 1
                if 0 <= idx < len(users):
                    user = users[idx]
                    confirm = input(f"Delete '{user.name}' and all data? (yes/no): ")
                    if confirm.lower() == 'yes':
                        manager.delete_user(user.user_id, delete_data=True)
            except ValueError:
                print("Invalid input.")
        
        elif choice == '4':
            break


if __name__ == "__main__":
    # Test user selection
    selector = UserSelector()
    user_id = selector.select_user()
    print(f"\n✓ Selected user ID: {user_id}")
