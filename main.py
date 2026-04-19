import os

def menu():
    print("\n===== FACE AUTH SYSTEM =====")
    print("1. Register User")
    print("2. Generate Embeddings")
    print("3. Start Authentication")
    print("4. Exit")

while True:
    menu()
    choice = input("Enter choice: ")

    if choice == "1":
        from src.register_user import register_user
        name = input("Enter name: ")
        user_class = input("Enter class: ")
        register_user(name, user_class)

    elif choice == "2":
        os.system("python src/generate_embeddings.py")

    elif choice == "3":
        os.system("python src/recognize.py")

    elif choice == "4":
        print("Exiting...")
        break

    else:
        print("Invalid choice")