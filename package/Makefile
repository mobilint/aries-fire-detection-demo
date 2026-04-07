##################     Setting    ##################
OPENCV_INCLUDE = `pkg-config --cflags opencv4`
OPENCV_LDFLAGS = `pkg-config --libs opencv4`
QBRUNTIME_INCLUDE = -I./qbruntime/include
QBRUNTIME_LDFLAGS = -L./qbruntime/lib -lqbruntime -Wl,-rpath=../qbruntime/lib
YAML_INCLUDE = -I./yaml-cpp/include
YAML_LDFLAGS = -L./yaml-cpp/lib -lyaml-cpp

LDFLAGS = $(OPENCV_LDFLAGS) $(QBRUNTIME_LDFLAGS) $(YAML_LDFLAGS)
INCLUDE = $(OPENCV_INCLUDE) $(QBRUNTIME_INCLUDE) $(YAML_INCLUDE)

# 소스 파일 디렉토리
SRC_DIR = ./src/demo
# 빌드 결과 디렉토리
BUILD_DIR = ./build
# 빌드 결과물 이름
TARGET_NAME = demo
# 오브젝트 파일 디렉토리
OBJ_DIR = ./obj
###################################################

TARGET = $(BUILD_DIR)/$(TARGET_NAME)

CXXFLAGS = -O3
LDFLAGS += -pthread -fopenmp
INCLUDE += -I./src

SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cc))
OBJS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(SRCS:.cc=.o))

.PHONY: all clean
all: $(TARGET)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(TARGET) : $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BUILD_DIR)
