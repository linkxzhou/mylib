// ES5: this binding
var obj = {x: 7, getX: function() { return this.x; }};
if (obj.getX() !== 7) throw 'fail';
