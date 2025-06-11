package main

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// App struct
type App struct {
	ctx context.Context
	objectCount int
	lastCommand string
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{
		objectCount: 0,
		lastCommand: "",
	}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	rand.Seed(time.Now().UnixNano())
}

// Greet returns a greeting for the given name
func (a *App) Greet(name string) string {
	return fmt.Sprintf("Hello %s, It's show time!", name)
}

// ProcessCommand 处理来自前端的命令
func (a *App) ProcessCommand(command string) string {
	a.lastCommand = command
	lowerCommand := strings.ToLower(strings.TrimSpace(command))
	
	switch {
	case strings.Contains(lowerCommand, "cube") || strings.Contains(lowerCommand, "立方体"):
		a.objectCount++
		return fmt.Sprintf("✅ 已添加立方体 #%d - 当前场景有 %d 个对象", a.objectCount, a.objectCount)
		
	case strings.Contains(lowerCommand, "sphere") || strings.Contains(lowerCommand, "球") || strings.Contains(lowerCommand, "球体"):
		a.objectCount++
		return fmt.Sprintf("✅ 已添加球体 #%d - 当前场景有 %d 个对象", a.objectCount, a.objectCount)
		
	case strings.Contains(lowerCommand, "clear") || strings.Contains(lowerCommand, "清空") || strings.Contains(lowerCommand, "删除"):
		a.objectCount = 0
		return "🗑️ 场景已清空，所有对象已移除"
		
	case strings.Contains(lowerCommand, "reset") || strings.Contains(lowerCommand, "重置"):
		return "🔄 相机视角已重置到初始位置"
		
	case strings.Contains(lowerCommand, "stop") || strings.Contains(lowerCommand, "停止"):
		return "⏸️ 动画已停止"
		
	case strings.Contains(lowerCommand, "start") || strings.Contains(lowerCommand, "开始"):
		return "▶️ 动画已开始"
		
	case strings.Contains(lowerCommand, "help") || strings.Contains(lowerCommand, "帮助"):
		return a.getHelpMessage()
		
	case strings.Contains(lowerCommand, "status") || strings.Contains(lowerCommand, "状态"):
		return a.getStatusMessage()
		
	case strings.Contains(lowerCommand, "random") || strings.Contains(lowerCommand, "随机"):
		return a.generateRandomCommand()
		
	default:
		return fmt.Sprintf("🤖 收到消息: '%s' - 尝试输入 'help' 查看可用命令", command)
	}
}

// getHelpMessage 返回帮助信息
func (a *App) getHelpMessage() string {
	return `📖 Web3D 客户端命令帮助:

🎮 场景控制:
• cube/立方体 - 添加立方体
• sphere/球体 - 添加球体
• clear/清空 - 清空场景
• reset/重置 - 重置相机

🎬 动画控制:
• start/开始 - 开始动画
• stop/停止 - 停止动画

📊 信息查询:
• status/状态 - 查看当前状态
• help/帮助 - 显示此帮助
• random/随机 - 生成随机命令

⌨️ 快捷键:
C-立方体, S-球体, X-清空, R-重置, 空格-切换动画`
}

// getStatusMessage 返回当前状态信息
func (a *App) getStatusMessage() string {
	return fmt.Sprintf(`📊 Web3D 客户端状态:

🎯 对象数量: %d
⏰ 当前时间: %s
💬 最后命令: %s
🚀 系统状态: 运行正常
🎮 引擎: Three.js + Wails v2
🖥️ 平台: Go + JavaScript`,
		a.objectCount,
		time.Now().Format("2006-01-02 15:04:05"),
		a.lastCommand)
}

// generateRandomCommand 生成随机命令建议
func (a *App) generateRandomCommand() string {
	commands := []string{
		"添加一个彩色立方体",
		"创建一个浮动球体",
		"清空当前场景",
		"重置相机视角",
		"开始动画效果",
		"停止所有动画",
	}
	
	randomIndex := rand.Intn(len(commands))
	return fmt.Sprintf("🎲 随机建议: %s", commands[randomIndex])
}

// GetObjectCount 获取当前对象数量
func (a *App) GetObjectCount() int {
	return a.objectCount
}

// SetObjectCount 设置对象数量（用于前端同步）
func (a *App) SetObjectCount(count int) {
	a.objectCount = count
}
