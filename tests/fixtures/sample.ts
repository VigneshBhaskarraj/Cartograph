import { z } from "./util";

export class Animal {
  speak(): string {
    return "...";
  }
}

export class Dog extends Animal {
  speak(): string {
    return bark();
  }
}

export function bark(): string {
  return "woof";
}

export const greet = (name: string): string => bark();

interface Named {
  name: string;
}
