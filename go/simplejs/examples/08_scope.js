// ES5: Function scope
var x = 1;
function foo() { var x = 2; if (x !== 2) throw 'fail'; }
foo();
if (x !== 1) throw 'fail';
