// Plain JavaScript fixture: classes, methods, arrow consts, imports, calls.
import { helper } from './util';

class Animal {
  speak() {
    return bark();
  }
}

class Dog extends Animal {
  speak() {
    return this.speak2();
  }
  speak2() {
    return "woof";
  }
}

function bark() {
  return helper("woof");
}

const greet = (name) => {
  return bark() + name;
};
