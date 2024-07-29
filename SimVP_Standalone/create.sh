#!/bin/bash

usage() {
    echo "Usage: $0 [--simulation simulation] [--num_initials num_initials] [--image_height image_height] [--image_width image_width] [--array_type array_type] [--chance chance] [--static_cells_random] [--dynamic_cells_random] [--increment increment] [--total_sample_length total_sample_length] [--normalize] [--num_samples num_samples] [--sample_start_index sample_start_index] [--total_length total_length] [--train_ratio train_ratio] [--val_ratio val_ratio] [--num_final_samples num_final_samples] [--datafolder datafolder]"
    echo "  --simulation             Type - str, Choices - [Simulation]. **Required**. Determines the simulation type."
    echo "  --num_initials           Type - int. **Required**. Specifies the number of initials."
    echo "  --image_height           Type - int. Specifies the image height."
    echo "  --image_width            Type - int. Specifies the image width."
    echo "  --array_type             Type - ArrayType. Defines the array type."
    echo "  --chance                 Type - float. Sets the chance parameter."
    echo "  --static_cells_random    Action - store_true. Only applies to HeatTransfer simulation, sets if mask cells should be random values."
    echo "  --dynamic_cells_random   Action - store_true. Only applies to HeatTransfer simulation, sets if non-mask cells should be random values."
    echo "  --increment increment    Type - int. Only applies to Boiling simulation, sets the increment value."
    echo "  --total_sample_length    Type - int. **Required**. Number of total iterations in the sequence including the initial state."
    echo "  --normalize              Action - store_true. Determines whether to normalize the data."
    echo "  --num_samples            Type - int. Specifies the number of samples."
    echo "  --sample_start_index     Type - int. Specifies the starting index of the sequence to be used for input."
    echo "  --total_length           Type - int. Specifies the total length."
    echo "  --train_ratio train_ratio  Type - float. Specifies the training data ratio."
    echo "  --val_ratio val_ratio      Type - float. Specifies the validation data ratio."
    echo "  --num_final_samples      Type - int. Specifies the number of final samples."
    echo "  --datafolder             Type - str. **Required**. Specifies the data folder path."
    exit 1
}

# Ensure DATAFOLDER is provided as an argument
if [ -z "$1" ]; then
    usage
else
    DATAFOLDER=""
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --simulation)
            SIMULATION="$2"
            shift 2
            ;;
        --num_initials)
            NUM_INITIALS="$2"
            shift 2
            ;;
        --image_height)
            IMAGE_HEIGHT="$2"
            shift 2
            ;;
        --image_width)
            IMAGE_WIDTH="$2"
            shift 2
            ;;
        --array_type)
            ARRAY_TYPE="$2"
            shift 2
            ;;
        --chance)
            CHANCE="$2"
            shift 2
            ;;
        --static_cells_random)
            STATIC_CELLS_RANDOM=true
            shift 1
            ;;
        --dynamic_cells_random)
            DYNAMIC_CELLS_RANDOM=true
            shift 1
            ;;
        --increment)
            INCREMENT="$2"
            shift 2
            ;;
        --total_sample_length)
            TOTAL_SAMPLE_LENGTH="$2"
            shift 2
            ;;
        --normalize)
            NORMALIZE=true
            shift 1
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --sample_start_index)
            SAMPLE_START_INDEX="$2"
            shift 2
            ;;
        --total_length)
            TOTAL_LENGTH="$2"
            shift 2
            ;;
        --train_ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --val_ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --num_final_samples)
            NUM_FINAL_SAMPLES="$2"
            shift 2
            ;;
        --datafolder)
            DATAFOLDER="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$DATAFOLDER" ]; then
    usage
fi

echo "Running with arguments:"
echo "  Datafolder: $DATAFOLDER"
echo "  Simulation: $SIMULATION"
echo "  Number of initials: $NUM_INITIALS"
echo "  Image height: $IMAGE_HEIGHT"
echo "  Image width: $IMAGE_WIDTH"
echo "  Array type: $ARRAY_TYPE"
echo "  Chance: $CHANCE"
echo "  Static cells random: $STATIC_CELLS_RANDOM"
echo "  Dynamic cells random: $DYNAMIC_CELLS_RANDOM"
echo "  Increment: $INCREMENT"
echo "  Total sample length: $TOTAL_SAMPLE_LENGTH"
echo "  Normalize: $NORMALIZE"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Sample start index: $SAMPLE_START_INDEX"
echo "  Total length: $TOTAL_LENGTH"
echo "  Train ratio: $TRAIN_RATIO"
echo "  Validation ratio: $VAL_RATIO"
echo "  Number of final samples: $NUM_FINAL_SAMPLES"

commands=(
    "python ./SimVP_Standalone/gen_initials.py \
        --simulation $SIMULATION \
        ${NUM_INITIALS:+--num_initials $NUM_INITIALS} \
        ${IMAGE_HEIGHT:+--image_height $IMAGE_HEIGHT} \
        ${IMAGE_WIDTH:+--image_width $IMAGE_WIDTH} \
        ${ARRAY_TYPE:+--array_type $ARRAY_TYPE} \
        ${CHANCE:+--chance $CHANCE} \
        ${STATIC_CELLS_RANDOM:+--static_cells_random} \
        ${DYNAMIC_CELLS_RANDOM:+--dynamic_cells_random} \
        --datafolder $DATAFOLDER"

    "python ./SimVP_Standalone/gen_samples.py \
        ${TOTAL_SAMPLE_LENGTH:+--total_sample_length $TOTAL_SAMPLE_LENGTH} \
        ${INCREMENT:+--increment $INCREMENT} \
        ${NORMALIZE:+--normalize} \
        --datafolder $DATAFOLDER"

    "python ./SimVP_Standalone/prep_loaders.py \
        ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES} \
        ${SAMPLE_START_INDEX:+--sample_start_index $SAMPLE_START_INDEX} \
        ${TOTAL_LENGTH:+--total_length $TOTAL_LENGTH} \
        ${TRAIN_RATIO:+--train_ratio $TRAIN_RATIO} \
        ${VAL_RATIO:+--val_ratio $VAL_RATIO} \
        --datafolder $DATAFOLDER"
)

for cmd in "${commands[@]}"; do
    echo "$cmd"
    eval "$cmd"
done
