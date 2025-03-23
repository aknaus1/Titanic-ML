# Titanic Passenger Survival Prediction

This project implements a Titanic passenger survival prediction model using Go. It reads passenger data from a CSV file and predicts whether each passenger survived based on various attributes.

## Project Structure

```
titanic-prediction
├── cmd
│   └── main.go          # Entry point of the application
├── internal
│   ├── model
│   │   └── titanic_model.go  # Defines the Titanic prediction model
│   └── passenger
│       └── passenger.go      # Defines the Passenger struct
├── internal
│   └── data
│       ├── train.csv         # Training data
│       ├── test.csv          # Test data
│       └── gender_submission.csv # Submission data example if only gender was considered
├── go.mod                # Go module definition
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd titanic-prediction
   ```

2. Initialize the Go module:
   ```
   go mod tidy
   ```

3. Ensure you have Go installed on your machine. You can download it from the official Go website.

## Usage

To run the application, use the following command:

```
go run cmd/main.go
```

## Prediction Logic

The prediction model uses various attributes of the passengers, such as age, gender, class, and fare, to determine the likelihood of survival. The current implementation includes a basic prediction logic that can be further enhanced with more sophisticated algorithms.

## Test Data

The test data is loaded from `internal/data/test.csv`.

## References

This project and datasets are based on the Kaggle competition "Titanic: Machine Learning from Disaster". You can find more information and the original datasets on the [Competition page](https://www.kaggle.com/competitions/titanic).