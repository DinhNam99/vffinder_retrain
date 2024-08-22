#include <stdio.h>
#include <string.h>

void secure_function(char *input) {
    char buffer[10];
    // Sử dụng strncpy để giới hạn độ dài sao chép
    strncpy(buffer, input, sizeof(buffer) - 1);
    // Đảm bảo buffer luôn kết thúc bằng ký tự null
    buffer[sizeof(buffer) - 1] = '\0';
    printf("Buffer content: %s\n", buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        secure_function(argv[1]);
    } else {
        printf("Please provide an input argument.\n");
    }
    return 0;
}
