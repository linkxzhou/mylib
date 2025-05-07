package simplejs

import (
	"errors"
	"fmt"
	"strings"
)

// Exception and error types
var ErrBreak = errors.New("break")

type JSException struct {
	Value JSValue
}

func (e *JSException) Error() string {
	return e.Value.ToString()
}

// JSArray represents an array value in JavaScript
type JSArray struct {
	Elements []JSValue
}

func (a *JSArray) ToString() string {
	elements := make([]string, 0, len(a.Elements))
	for _, v := range a.Elements {
		elements = append(elements, v.ToString())
	}
	return fmt.Sprintf("[%s]", strings.Join(elements, ", "))
}
