package register

import (
	"fmt"
	"strings"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/importer"
)

var kindName = map[importer.BasicKind]string{
	importer.Var:             "Variable",
	importer.Const:           "Constant",
	importer.TypeName:        "Struct",
	importer.Function:        "Function",
	importer.BuiltinFunction: "Function",
}

// KeywordInfo 代码补全的提示信息
type KeywordInfo struct {
	Label           string `json:"label"`
	Kind            string `json:"kind"`
	InsertText      string `json:"insertText"`
	InsertTextRules string `json:"insertTextRules"`
}

// Keywords 获取当前已导入包的关键字信息，用于做代码补全
func Keywords() []*KeywordInfo {
	keywords := make([]*KeywordInfo, 0)
	for _, pkg := range importer.GetAllPackages() {
		for _, object := range pkg.Objects {
			info := KeywordInfo{
				Label:           fmt.Sprintf("%s.%s", pkg.Name, object.Name),
				Kind:            kindName[object.Kind],
				InsertText:      "",
				InsertTextRules: "",
			}

			if info.Kind == "Function" {
				inParam := make([]string, 0)
				for i := 0; i < object.Type.NumIn(); i++ {
					inParam = append(inParam, fmt.Sprintf("${%d:%s}", i+1, object.Type.In(i).String()))
				}
				info.InsertText = fmt.Sprintf("%s(%s)", info.Label, strings.Join(inParam, ","))
				info.InsertTextRules = "InsertAsSnippet"
			} else {
				info.InsertText = pkg.Name + "." + object.Name
			}
			keywords = append(keywords, &info)
		}
	}

	return keywords
}
