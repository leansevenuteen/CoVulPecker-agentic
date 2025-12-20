// Example vulnerable C code for testing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Vulnerability 1: Buffer Overflow
void vulnerable_copy(char *user_input) {
    char buffer[64];
    strcpy(buffer, user_input);  // No bounds checking!
    printf("Copied: %s\n", buffer);
}

// Vulnerability 2: Format String Vulnerability
void vulnerable_print(char *user_input) {
    printf(user_input);  // Format string vulnerability!
}

// Vulnerability 3: Use After Free
void use_after_free() {
    char *ptr = (char *)malloc(100);
    strcpy(ptr, "Hello World");
    free(ptr);
    printf("%s\n", ptr);  // Use after free!
}

// Vulnerability 4: Integer Overflow
void integer_overflow(int size) {
    if (size < 0) return;
    
    // Integer overflow when size is very large
    int buffer_size = size + 100;
    char *buffer = (char *)malloc(buffer_size);
    
    if (buffer == NULL) return;
    
    // This could write beyond allocated memory
    memset(buffer, 'A', size);
    free(buffer);
}

// Vulnerability 5: SQL Injection (simulated)
void execute_query(char *username) {
    char query[256];
    // SQL Injection vulnerability
    sprintf(query, "SELECT * FROM users WHERE name = '%s'", username);
    printf("Executing: %s\n", query);
    // Assume this executes the query...
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input>\n", argv[0]);
        return 1;
    }
    
    vulnerable_copy(argv[1]);
    vulnerable_print(argv[1]);
    
    return 0;
}

