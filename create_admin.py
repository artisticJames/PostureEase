from db import create_user
import bcrypt

def create_admin_user():
    # Admin user details
    admin_data = {
        'username': 'admin',
        'email': 'admin@posturease.com',
        'password': 'admin123',
        'first_name': 'Admin',
        'last_name': 'User',
        'birth_date': '2000-01-01',
        'gender': 'other'
    }
    
    # Create admin user
    success, result = create_user(**admin_data)
    
    if success:
        print("Admin user created successfully!")
        print(f"Username: {admin_data['username']}")
        print(f"Password: {admin_data['password']}")
    else:
        print("Failed to create admin user:")
        print(result)

if __name__ == "__main__":
    create_admin_user() 