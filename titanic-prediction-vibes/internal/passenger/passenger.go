// passenger.go
package passenger

// Passenger represents a passenger on the Titanic
type Passenger struct {
	PassengerId int
	Survived    int
	Pclass      int
	Name        string
	Gender      string
	Age         float64
	SibSp       int
	Parch       int
	Ticket      string
	Fare        float64
	Cabin       string
	Embarked    string
}
