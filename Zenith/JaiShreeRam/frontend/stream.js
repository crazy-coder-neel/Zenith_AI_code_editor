// Welcome to AI Code Editor
// This is a sample JavaScript file

function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n) {
    if (n === 0) return 1;
    return n * factorial(n - 1);
}

// Example usage
console.log('Fibonacci of 10:', fibonacci(10));
console.log('Factorial of 5:', factorial(5));