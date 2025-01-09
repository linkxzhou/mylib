package gofun

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/value"
)

func binopADD(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.String:
		result = x.String() + y.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() + y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() + y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() + y.Uint()
	}
	return result
}

func binopSUB(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() - y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() - y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() - y.Uint()
	}
	return result
}

func binopMUL(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() * y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() * y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() * y.Uint()
	}
	return result
}

func binopQUO(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() / y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() / y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() / y.Uint()
	}
	return result
}

func binopREM(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() % y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() % y.Uint()
	}
	return result
}

func binopAND(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() & y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() & y.Uint()
	}
	return result
}

func binopOR(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() | y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() | y.Uint()
	}
	return result
}

func binopXOR(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() ^ y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() ^ y.Uint()
	}
	return result
}

func binopANDNOT(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() &^ y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() &^ y.Uint()
	}
	return result
}

func binopSHL(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() << y.Uint()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() << y.Uint()
	}
	return result
}

func binopSHR(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() >> y.Uint()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() >> y.Uint()
	}
	return result
}

func binopLSS(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.String:
		result = x.String() < y.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() < y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() < y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() < y.Uint()
	}
	return result
}

func binopLEQ(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.String:
		result = x.String() <= y.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() <= y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() <= y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() <= y.Uint()
	}
	return result
}

func binopEQL(x, y value.Value) interface{} {
	var result interface{}
	if x.IsNil() || y.IsNil() {
		result = x.IsNil() && y.IsNil()
	} else {
		result = x.Interface() == y.Interface()
	}
	return result
}

func binopNEQ(x, y value.Value) interface{} {
	var result interface{}
	if x.IsNil() || y.IsNil() {
		result = x.IsNil() != y.IsNil()
	} else {
		result = x.Interface() != y.Interface()
	}
	return result
}

func binopGTR(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.String:
		result = x.String() > y.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() > y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() > y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() > y.Uint()
	}
	return result
}

func binopGEQ(x, y value.Value) interface{} {
	var result interface{}
	switch x.Kind() {
	case reflect.String:
		result = x.String() >= y.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		result = x.Int() >= y.Int()
	case reflect.Float32, reflect.Float64:
		result = x.Float() >= y.Float()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		result = x.Uint() >= y.Uint()
	}
	return result
}
