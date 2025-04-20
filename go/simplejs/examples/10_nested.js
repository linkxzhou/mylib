// ES5: Nested functions and closures
function outer(a) {
  return function(b) { return a + b; };
}
var fn = outer(3);
if (fn(4) !== 7) throw 'fail';
