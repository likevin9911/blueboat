#!/bin/bash
# Script to generate world file from xacro before launching Gazebo

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse arguments (roslaunch passes them as arg:=value)
WAVE_MODEL="PMS"
WAVE_PERIOD="5"
WAVE_NUMBER="3"
WAVE_SCALE="2.5"
WAVE_GAIN="0.2"
WAVE_DIRECTION="1.0 0.0"
WAVE_ANGLE="0.4"
WAVE_TAU="2.0"
WAVE_AMPLITUDE="0.5"
WAVE_STEEPNESS="0.0"
WIND_DIR_DEG="270"
WIND_MEAN="0"
WIND_VAR_GAIN="0"
WIND_TAU="2"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        wave_model:=*) WAVE_MODEL="${arg#*:=}" ;;
        wave_period:=*) WAVE_PERIOD="${arg#*:=}" ;;
        wave_number:=*) WAVE_NUMBER="${arg#*:=}" ;;
        wave_scale:=*) WAVE_SCALE="${arg#*:=}" ;;
        wave_gain:=*) WAVE_GAIN="${arg#*:=}" ;;
        wave_direction:=*) WAVE_DIRECTION="${arg#*:=}" ;;
        wave_angle:=*) WAVE_ANGLE="${arg#*:=}" ;;
        wave_tau:=*) WAVE_TAU="${arg#*:=}" ;;
        wave_amplitude:=*) WAVE_AMPLITUDE="${arg#*:=}" ;;
        wave_steepness:=*) WAVE_STEEPNESS="${arg#*:=}" ;;
        wind_dir_deg:=*) WIND_DIR_DEG="${arg#*:=}" ;;
        wind_mean:=*) WIND_MEAN="${arg#*:=}" ;;
        wind_var_gain:=*) WIND_VAR_GAIN="${arg#*:=}" ;;
        wind_tau:=*) WIND_TAU="${arg#*:=}" ;;
    esac
done

# Find the package path
PACKAGE_PATH=$(rospack find blueboat)
XACRO_FILE="$PACKAGE_PATH/worlds/open_ocean.world.xacro"
OUTPUT_FILE="$PACKAGE_PATH/worlds/open_ocean.generated.world"

echo "Generating world file with parameters:"
echo "  Wave model: $WAVE_MODEL"
echo "  Wave amplitude: $WAVE_AMPLITUDE"
echo "  Wind mean: $WIND_MEAN"
echo "  Output: $OUTPUT_FILE"

# Generate the world file
rosrun xacro xacro "$XACRO_FILE" \
    wave_model:="$WAVE_MODEL" \
    wave_period:="$WAVE_PERIOD" \
    wave_number:="$WAVE_NUMBER" \
    wave_scale:="$WAVE_SCALE" \
    wave_gain:="$WAVE_GAIN" \
    wave_direction:="$WAVE_DIRECTION" \
    wave_angle:="$WAVE_ANGLE" \
    wave_tau:="$WAVE_TAU" \
    wave_amplitude:="$WAVE_AMPLITUDE" \
    wave_steepness:="$WAVE_STEEPNESS" \
    wind_dir_deg:="$WIND_DIR_DEG" \
    wind_mean:="$WIND_MEAN" \
    wind_var_gain:="$WIND_VAR_GAIN" \
    wind_tau:="$WIND_TAU" \
    -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "World file generated successfully!"
else
    echo "ERROR: Failed to generate world file"
    exit 1
fi
