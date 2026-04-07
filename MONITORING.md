### load average
top -H -p <PID>

### 초당 컨텍스트 스위치 양
pidstat -w -p <PID> 1

### 스레드 별 컨텍스트 스위치
pidstat -wt -p <PID> 1

### cpu가 실행하는 코드 추적
sudo perf record -F 99 -g -p <PID> -- sleep 20
sudo perf report