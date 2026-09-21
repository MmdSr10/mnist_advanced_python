# Advanced MNIST Classification Pipeline 🚀

A production-ready, highly structured, and object-oriented implementation of the classic MNIST digit classification problem. This project serves as a showcase of advanced Python software engineering practices applied to Deep Learning pipelines using TensorFlow and Keras.

## 🎯 Project Focus
While solving MNIST is trivial, building a maintainable, scalable, and robust machine learning pipeline is not. This repository moves away from standard Jupyter Notebook scripts and demonstrates how to structure an AI project for production environments. 

Tested and optimized for Linux environments (Ubuntu 24) and accelerated on NVIDIA GPUs (e.g., RTX 5060 series).

## ✨ Advanced Python Concepts Implemented

This codebase is heavily refactored to utilize modern Python and Software Engineering best practices:

- **Object-Oriented Programming (OOP):** Modular design separating Data Loading, Model Architecture, Training, and Evaluation.
- **Dataclasses & Strong Typing:** Centralized configuration management using `@dataclass(frozen=True)` and strict Python Type Hinting (`typing` module) for predictable inputs/outputs.
- **Custom Exceptions:** Domain-specific error handling (e.g., `DataLoadError`, `ModelBuildError`) instead of generic exceptions for faster debugging.
- **Advanced Logging:** A centralized logging system replacing raw `print()` statements, outputting formatted logs to both the console and a persistent `logs/` directory.
- **Context Managers (`with`):** Custom context managers to safely handle resource allocation and log execution times during data processing.
- **Generators (`yield`):** Memory-efficient custom batch generators for the training loop, demonstrating how to handle datasets that exceed system RAM.
- **Decorators (`@`):** Custom function wrappers (e.g., `@log_build_time`) to dynamically inject benchmarking and logging capabilities into model compilation steps.

## 📂 Project Structure

```text
mnist_advanced/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point & orchestration
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Generators & Context Managers for data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py        # OOP model architecture & Decorators
│   │   ├── train.py            # Custom training loop
│   │   └── evaluate.py         # Evaluation and metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Dataclasses for hyperparameters
│       ├── exceptions.py       # Custom exception hierarchy
│       └── logger.py           # Centralized logging configuration
├── logs/                       # Auto-generated execution logs
├── saved_models/               # Serialized trained models (.keras)
├── .gitignore
├── requirements.txt
└── README.md