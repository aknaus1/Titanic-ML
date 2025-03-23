package model

import (
	"encoding/csv"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"titanic-prediction/internal/passenger"
)

// TitanicModel represents the Titanic prediction model
type TitanicModel struct {
	weights []float64
}

// NewTitanicModel initializes a new TitanicModel with given weights
func NewTitanicModel(weights []float64) *TitanicModel {
	return &TitanicModel{weights: weights}
}

// sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// PredictSurvival predicts whether a passenger survived based on their attributes
func (model *TitanicModel) PredictSurvival(p passenger.Passenger) int {
	// Feature vector: [1, Pclass, Gender, Age, SibSp, Parch, Fare, Embarked, Cabin, Title]
	gender := 0.0
	if p.Gender == "female" {
		gender = 1.0
	}

	embarked := 0.0
	if p.Embarked == "C" {
		embarked = 1.0
	} else if p.Embarked == "Q" {
		embarked = 2.0
	}

	cabin := 0.0
	if p.Cabin != "" {
		cabin = 1.0
	}

	title := extractTitle(p.Name)

	features := []float64{
		1, // Intercept term
		float64(p.Pclass),
		float64(gender),
		float64(p.Age),
		float64(p.SibSp),
		float64(p.Parch),
		float64(p.Fare),
		embarked,
		cabin,
		title,
	}

	// Normalize features
	features = normalize(features)

	// Compute the weighted sum
	z := 0.0
	for i, weight := range model.weights {
		z += weight * features[i]
	}

	// Apply the sigmoid function
	probability := sigmoid(z)

	// Predict survival if probability > 0.5
	// return probability > 0.5
	if probability > 0.50 {
		return 1
	} else {
		return 0
	}
}

// TrainModel trains a logistic regression model and returns the weights
func TrainModel(fileName string) []float64 {
	// Load the data
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	// Initialize variables
	var X [][]float64
	var y []float64

	// Process the data
	for _, record := range records[1:] { // Skip header line
		survived, _ := strconv.ParseFloat(record[1], 64) // Assuming the survival column is the first column
		pclass, _ := strconv.ParseFloat(record[2], 64)
		// gender: male (0) female (1)
		gender := 0.0
		if record[4] == "female" {
			gender = 1.0
		}
		age, _ := strconv.ParseFloat(record[5], 64)
		sibsp, _ := strconv.ParseFloat(record[6], 64)
		parch, _ := strconv.ParseFloat(record[7], 64)
		// ticket := record[8]
		fare, _ := strconv.ParseFloat(record[9], 64)
		// cabin: not empty (1) empty (0)
		cabin := 0.0
		if record[10] != "" {
			cabin = 1.0
		}
		// embarked: S (0), C (1), Q (2)
		embarked := 0.0
		if record[11] == "C" {
			embarked = 1.0
		} else if record[11] == "Q" {
			embarked = 2.0
		}
		// title extracted from name
		title := extractTitle(record[3])

		// Feature vector: [1, Pclass, Gender, Age, SibSp, Parch, Fare, Embarked, Cabin, Title]
		features := []float64{1, pclass, gender, age, sibsp, parch, fare, embarked, cabin, title}
		features = normalize(features)
		X = append(X, features)

		// Target variable (Survived)
		y = append(y, survived)
	}

	// Initialize weights
	weights := make([]float64, len(X[0]))

	// Train the model (simple gradient descent)
	learningRate := 0.001
	iterations := 1000000

	for iter := 0; iter < iterations; iter++ {
		gradients := make([]float64, len(weights))

		for i := 0; i < len(X); i++ {
			z := 0.0
			for j := 0; j < len(weights); j++ {
				z += weights[j] * X[i][j]
			}
			prediction := sigmoid(z)
			error := prediction - y[i]

			for j := 0; j < len(weights); j++ {
				gradients[j] += error * X[i][j]
			}
		}

		for j := 0; j < len(weights); j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
		}
	}

	return weights
}

// SaveWeights saves the model weights to a CSV file
func (model *TitanicModel) SaveWeights(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	record := make([]string, len(model.weights))
	for i, weight := range model.weights {
		record[i] = strconv.FormatFloat(weight, 'f', 6, 64)
	}

	if err := writer.Write(record); err != nil {
		return err
	}

	return nil
}

// LoadWeights loads the model weights from a CSV file
func LoadWeights(fileName string) ([]float64, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	record, err := reader.Read()
	if err != nil {
		return nil, err
	}

	weights := make([]float64, len(record))
	for i, value := range record {
		weights[i], err = strconv.ParseFloat(value, 64)
		if err != nil {
			return nil, err
		}
	}

	return weights, nil
}

// extractTitle extracts the title from the passenger's name
func extractTitle(name string) float64 {
	if strings.Contains(name, "Mr.") {
		return 1.0
	} else if strings.Contains(name, "Mrs.") {
		return 2.0
	} else if strings.Contains(name, "Miss.") {
		return 3.0
	} else if strings.Contains(name, "Ms.") {
		return 4.0
	} else if strings.Contains(name, "Master.") {
		return 5.0
	} else if strings.Contains(name, "Dr.") {
		return 6.0
	} else if strings.Contains(name, "Rev.") {
		return 7.0
	} else if strings.Contains(name, "Col.") {
		return 8.0
	} else if strings.Contains(name, "Major.") {
		return 9.0
	} else if strings.Contains(name, "Capt.") {
		return 10.0
	} else if strings.Contains(name, "Mlle.") {
		return 11.0
	} else if strings.Contains(name, "Mme.") {
		return 12.0
	} else if strings.Contains(name, "Sir.") {
		return 13.0
	} else if strings.Contains(name, "Lady.") {
		return 14.0
	} else if strings.Contains(name, "Countess.") {
		return 15.0
	} else if strings.Contains(name, "Jonkheer.") {
		return 16.0
	} else if strings.Contains(name, "Don.") {
		return 17.0
	}
	log.Printf("No title found for name: %s", name)
	return 0.0
}

// normalize normalizes the feature vector
func normalize(features []float64) []float64 {
	mean := make([]float64, len(features))
	std := make([]float64, len(features))

	for i := 0; i < len(features); i++ {
		mean[i] = 0.0
		std[i] = 0.0
	}

	for i := 0; i < len(features); i++ {
		mean[i] += features[i]
	}

	for i := 0; i < len(features); i++ {
		mean[i] /= float64(len(features))
	}

	for i := 0; i < len(features); i++ {
		std[i] += (features[i] - mean[i]) * (features[i] - mean[i])
	}

	for i := 0; i < len(features); i++ {
		std[i] = math.Sqrt(std[i] / float64(len(features)))
	}

	for i := 0; i < len(features); i++ {
		if std[i] != 0 {
			features[i] = (features[i] - mean[i]) / std[i]
		}
	}

	return features
}
