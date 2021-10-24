#include <stdio.h>
 #include <time.h>
 
 int main() {
         time_t now = time(NULL);
         struct tm* local = localtime(&now);
         printf("Chúc mừng năm mới %d!", local->tm_year + 1900);
         return 0;
 }