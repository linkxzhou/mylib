// ES5: for loop
var sum = 0;
for (var i = 0; i < 4; i = i + 1) { sum = sum + i; }
if (sum !== 6) throw 'fail';
