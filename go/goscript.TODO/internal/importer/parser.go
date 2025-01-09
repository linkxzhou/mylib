package importer

import (
	"go/token"
	"go/types"
	"reflect"
)

func (p *Importer) parseNameType(t reflect.Type) (named *types.Named) {
	pkg := p.Package(t.PkgPath())
	name := t.Name()
	if pkg != nil {
		scope := pkg.Scope()
		if obj := scope.Lookup(name); obj == nil {
			typeName := types.NewTypeName(token.NoPos, pkg, name, nil)
			named = types.NewNamed(typeName, nil, nil)
			scope.Insert(typeName)
			obj = typeName
		} else {
			named = obj.Type().(*types.Named)
		}
	} else {
		typeName := types.NewTypeName(token.NoPos, pkg, name, nil)
		named = types.NewNamed(typeName, nil, nil)
	}
	p.typeCache[t] = named
	return named
}
