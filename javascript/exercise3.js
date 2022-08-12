function f(x) {
    return x ** 2;
}

console.log(f(3));

function f(x, y) {
    return x + y;
}

console.log(f(1, 2));

function add(x, y) {
    if (typeof x != 'number' || typeof y != 'number'){
        throw new TypeError('');
    }
    return x + y;
}

function add2(x=0, y=0) {
    if (typeof x != 'number' || typeof y != 'number'){
        throw new TypeError('');
    }
    return x + y;
}

function add3() {
    return arguments;
}

function add4(...numbers) {
    return numbers;
}

var a = [1, 2, 3];

console.log(a[0]);

console.log(a.length);

delete a[0];

a[0] = 1;

a.splice(0, 1);

var obj = {
    name: 'dan',
    age: 22
};

for (var key in obj) {
    console.log(`${key} : ${obj[key]}`);
}

var arr = [1, 2, 3]

for (var idx in arr) {
    console.log(idx, arr[idx]);
}

