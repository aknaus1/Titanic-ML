package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"titanic-prediction/internal/model"
	"titanic-prediction/internal/passenger"
)

func loadTestData(testFileName, submissionFileName string) ([]passenger.Passenger, []int) {
	// Load test data
	testFile, err := os.Open(testFileName)
	if err != nil {
		log.Fatalf("failed to open test file: %v", err)
	}
	defer testFile.Close()

	testReader := csv.NewReader(testFile)
	testRecords, err := testReader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read test CSV: %v", err)
	}

	// Load submission data
	submissionFile, err := os.Open(submissionFileName)
	if err != nil {
		log.Fatalf("failed to open submission file: %v", err)
	}
	defer submissionFile.Close()

	submissionReader := csv.NewReader(submissionFile)
	submissionRecords, err := submissionReader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read submission CSV: %v", err)
	}

	// Map PassengerId to Survived
	survivalMap := make(map[int]int)
	for _, record := range submissionRecords[1:] { // Skip header line
		passengerId, _ := strconv.Atoi(record[0])
		survived, _ := strconv.Atoi(record[1])
		survivalMap[passengerId] = survived
	}

	var passengers []passenger.Passenger
	var actualSurvival []int

	for _, record := range testRecords[1:] { // Skip header line
		if len(record) < 11 {
			continue // Skip records with insufficient data
		}
		passengerId, _ := strconv.Atoi(record[0])
		pclass, _ := strconv.Atoi(record[1])
		age, _ := strconv.ParseFloat(record[4], 64)
		sibsp, _ := strconv.Atoi(record[5])
		parch, _ := strconv.Atoi(record[6])
		fare, _ := strconv.ParseFloat(record[8], 64)

		passenger := passenger.Passenger{
			PassengerId: passengerId,
			Pclass:      pclass,
			Name:        record[2],
			Gender:      record[3],
			Age:         age,
			SibSp:       sibsp,
			Parch:       parch,
			Ticket:      record[7],
			Fare:        fare,
			Cabin:       record[9],
			Embarked:    record[10],
		}
		passengers = append(passengers, passenger)
		actualSurvival = append(actualSurvival, survivalMap[passengerId])
	}

	return passengers, actualSurvival
}

func main() {
	weights := model.TrainModel("internal/data/train.csv")
	titanicModel := model.NewTitanicModel(weights)

	// Load test data
	testPassengers, actualSurvival := loadTestData("internal/data/test.csv", "internal/data/gender_submission.csv")

	// Make predictions and measure accuracy
	correctPredictions := 0
	for i, p := range testPassengers {
		predictedSurvival := titanicModel.PredictSurvival(p)
		if predictedSurvival == (actualSurvival[i] == 1) {
			correctPredictions++
		}
	}

	accuracy := float64(correctPredictions) / float64(len(testPassengers))
	fmt.Printf("Model accuracy: %.2f%%\n", accuracy*100)
}
