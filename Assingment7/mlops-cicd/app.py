from src.model import train_model


def main():
    accuracy = train_model()
    print("ML application started.")
    print(f"Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
