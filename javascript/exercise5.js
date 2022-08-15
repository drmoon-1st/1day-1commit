function aaa(name, age) {
    this.name = name;
    this.age = age;
}

let user = new aaa('bbb', 111);

console.log(user.name, user.age);

function user(name) {
    console.log(new.target);
    if (!new.target){
        return new user(name);
    }
}

user('a')
let user = new user('aa');

let test = (name, age) => {
    this.name = name;
    this.age = age;
}
test('aa0', 11);
console.log(test.name, test.age);

