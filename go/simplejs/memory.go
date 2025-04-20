package simplejs

import "errors"

// ErrOOM indicates out of memory.
var ErrOOM = errors.New("out of memory")

// Memory is a simple memory pool for JS engine.
type Memory struct {
	buf  []byte
	brk  int
	size int
}

// NewMemory creates a memory pool of given size.
func NewMemory(size int) *Memory {
	return &Memory{buf: make([]byte, size), size: size}
}

// align32 rounds n up to multiple of 4.
func align32(n int) int {
	if rem := n % 4; rem != 0 {
		return n + (4 - rem)
	}
	return n
}

// Alloc reserves n bytes and returns the offset.
func (m *Memory) Alloc(n int) (int, error) {
	n = align32(n)
	if m.brk+n > m.size {
		return 0, ErrOOM
	}
	offset := m.brk
	m.brk += n
	return offset, nil
}

// Write writes data into memory at given offset.
func (m *Memory) Write(off int, data []byte) {
	copy(m.buf[off:], data)
}

// Read reads length bytes from memory at given offset.
func (m *Memory) Read(off int, length int) []byte {
	return m.buf[off : off+length]
}
