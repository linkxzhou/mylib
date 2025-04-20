// ES5: Array literal and indexing
var arr = [1, 2, 3];
if (arr[0] !== 1 || arr[2] !== 3) throw 'fail';
arr[1] = 42;
if (arr[1] !== 42) throw 'fail';
