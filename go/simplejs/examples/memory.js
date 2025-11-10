// Memory management test
let objects = [];
for (let i = 0; i < 100; i = i + 1) {
  objects.push({index: i, value: 'test' + i});
}
objects.length
