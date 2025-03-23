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

func loadTestData(testFileName string) ([]passenger.Passenger, []int) {
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

	var passengers []passenger.Passenger
	var actualSurvival []int

	for _, record := range testRecords[1:] { // Skip header line
		if len(record) < 11 {
			continue // Skip records with insufficient data
		}
		passengerId, _ := strconv.Atoi(record[0])
		survival, _ := strconv.Atoi(record[1])
		pclass, _ := strconv.Atoi(record[2])
		age, _ := strconv.ParseFloat(record[5], 64)
		sibsp, _ := strconv.Atoi(record[6])
		parch, _ := strconv.Atoi(record[7])
		fare, _ := strconv.ParseFloat(record[9], 64)

		passenger := passenger.Passenger{
			PassengerId: passengerId,
			Pclass:      pclass,
			Name:        record[3],
			Gender:      record[4],
			Age:         age,
			SibSp:       sibsp,
			Parch:       parch,
			Ticket:      record[7],
			Fare:        fare,
			Cabin:       record[9],
			Embarked:    record[10],
		}
		passengers = append(passengers, passenger)
		actualSurvival = append(actualSurvival, survival)
	}

	return passengers, actualSurvival
}

func main() {
	var titanicModel *model.TitanicModel
	if len(os.Args) < 2 {
		// Try to load weights from file
		weights, err := model.LoadWeights("internal/data/weights.csv")
		if err != nil {
			log.Fatalf("failed to load weights: %v", err)
			return
		}
		titanicModel = model.NewTitanicModel(weights)
		fmt.Println("Loaded model from weights.csv")
	} else {
		trainFileName := os.Args[1]
		// Train the model
		weights := model.TrainModel(trainFileName)
		titanicModel = model.NewTitanicModel(weights)
		titanicModel.SaveWeights("internal/data/weights.csv")
		fmt.Println("Trained and saved model to weights.csv")
	}

	// Load test data
	testPassengers, actualSurvival := loadTestData("internal/data/train.csv")

	// Make predictions and measure accuracy
	correctPredictions := 0
	var predictions [][]string
	predictions = append(predictions, []string{"PassengerId", "Survived"})

	for i, p := range testPassengers {
		predictedSurvival := titanicModel.PredictSurvival(p)
		if predictedSurvival == actualSurvival[i] {
			correctPredictions++
		}
		predictions = append(predictions, []string{strconv.Itoa(p.PassengerId), strconv.Itoa(predictedSurvival)})
	}

	accuracy := float64(correctPredictions) / float64(len(testPassengers))
	fmt.Printf("Model accuracy: %.2f\n", accuracy)

	// Save predictions to CSV file
	outputFile, err := os.Create("cmd/predictions.csv")
	if err != nil {
		log.Fatalf("failed to create output file: %v", err)
	}
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	for _, record := range predictions {
		if err := writer.Write(record); err != nil {
			log.Fatalf("failed to write record to csv: %v", err)
		}
	}
}
