// const title = document.getElementById("title");

// title.innerText("got you");

// console.log("hewwwww");
// var a = 3;
// console.log(a);
// a = 2;
// console.log(a);

// let b = 3;
// console.log(b);
// b = 4;
// console.log(b);

// const v = 4;
// console.log(v)

// var은 사용 x

// var number = 1;
// console.log(typeof number, number)

// var string = '11';
// console.log(typeof string, string);

// var bool = true;
// console.log(typeof bool, bool);

// var un;
// console.log(typeof un);

// var a = null;
// console.log(typeof a, a);

// var sym = Symbol("sss");
// console.log(typeof sym, sym);

// var a = 'hi';
// var b = 'bye';
// console.log(`${a} $00{b}`)

// var obj = {};
// console.log(typeof obj);

// var a = 1;
// console.log(a);

// let b = "hi";
// console.log(b);

// const c = null;
// console.log(c);

// var d;
// console.log(typeof d);

// var e = true;
// console.log(e);

// var f = Symbol("key");
// console.log(f);

// var g = {};
// console.log(typeof g);

// var a = 1;
// console.log(a);
// a++
// console.log(a);
// a--
// console.log(a);

// var a = NaN;
// console.log(isNaN(a))

// var b = "12";
// console.log(12 === +b)

// console.log(0 == '')
// console.log(0 === '')

// {
//     console.log(11)
// }

// var a = 5;
// switch(a){
//     case 1:
//         console.log(1111)
//         break;
//     case 2:
//         console.log(3333)
//         break;
//     default:
//         console.log(3393939)
// }

// for (var i=0; i < 5; i=i+1){
//     console.log(i)
// }

// console.log(1 + '11')
// console.log(1 + true)

// console.log(typeof String(12))
// console.log(typeof true.toString())

// console.log(typeof Number('123'))
// console.log(parseInt('12'))

// var person = {
//     name: 'dan',
//     say: function() {
//         console.log(`my name is ${this.name}`)
//     }
// };
// console.log(person.name);
// person.say();

person = new Object();
person.name = 'dan';
person.say = function() {
    console.log(this.name)
};
person.say();