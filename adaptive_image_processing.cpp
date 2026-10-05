#include <mpi.h>
#include <omp.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

// Colour-grading modes selected by the adaptive analysis
enum ProcessingMode
{
    VIBRANT_NATURE = 0,
    WARM_BALANCE = 1,
    VIBRANT = 2,
    CINEMATIC = 3
};

// Stores the characteristics measured from the complete image
struct ImageAnalysis
{
    double averageBrightness;
    double contrast;
    double averageRed;
    double averageGreen;
    double averageBlue;
    ProcessingMode mode;
};

// OpenCL kernel that applies the selected colour grade
const char* kernelSource = R"CLC(
__kernel void processImage(
    __global const uchar* input,
    __global uchar* output,
    const int totalPixels,
    const int mode)
{
    int p = get_global_id(0);

    if (p >= totalPixels)
        return;

    int i = p * 3;

    float r = (float)input[i];
    float g = (float)input[i + 1];
    float b = (float)input[i + 2];

    float newR = r;
    float newG = g;
    float newB = b;

    if (mode == 0)
    {
        // VIBRANT NATURE
        // Strengthen natural greens while increasing saturation and contrast
        float gray =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        newR =
            gray + (r - gray) * 1.28f;

        newG =
            gray + (g - gray) * 1.48f;

        newB =
            gray + (b - gray) * 1.20f;

        newG *= 1.08f;

        newR =
            (newR - 110.0f) * 1.16f + 110.0f;

        newG =
            (newG - 110.0f) * 1.16f + 110.0f;

        newB =
            (newB - 110.0f) * 1.16f + 110.0f;
    }
    else if (mode == 1)
    {
        // WARM BALANCE
        // Correct blue-dominant scenes using a warmer colour balance
        newR =
            r * 1.20f + 12.0f;

        newG =
            g * 1.07f + 4.0f;

        newB =
            b * 0.82f;

        newR =
            (newR - 120.0f) * 1.08f + 120.0f;

        newG =
            (newG - 120.0f) * 1.08f + 120.0f;

        newB =
            (newB - 120.0f) * 1.08f + 120.0f;
    }
    else if (mode == 2)
    {
        // VIBRANT
        // Improve flat images using stronger saturation and contrast
        float gray =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        newR =
            gray + (r - gray) * 1.45f;

        newG =
            gray + (g - gray) * 1.45f;

        newB =
            gray + (b - gray) * 1.45f;

        newR =
            (newR - 128.0f) * 1.20f + 128.0f;

        newG =
            (newG - 128.0f) * 1.20f + 128.0f;

        newB =
            (newB - 128.0f) * 1.20f + 128.0f;
    }
    else if (mode == 3)
    {
        // CINEMATIC
        // Apply cool shadows and warm highlights
        float brightness =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        newR =
            (r - 128.0f) * 1.18f + 128.0f;

        newG =
            (g - 128.0f) * 1.14f + 128.0f;

        newB =
            (b - 128.0f) * 1.18f + 128.0f;

        if (brightness < 100.0f)
        {
            // Cooler teal shadows
            newR -= 12.0f;
            newG += 9.0f;
            newB += 18.0f;
        }
        else if (brightness > 165.0f)
        {
            // Warmer golden highlights
            newR += 18.0f;
            newG += 8.0f;
            newB -= 12.0f;
        }
        else
        {
            // Smaller colour shift through mid-tones
            newR += 4.0f;
            newG += 3.0f;
            newB += 2.0f;
        }
    }

    newR =
        clamp(newR, 0.0f, 255.0f);

    newG =
        clamp(newG, 0.0f, 255.0f);

    newB =
        clamp(newB, 0.0f, 255.0f);

    output[i] =
        (uchar)newR;

    output[i + 1] =
        (uchar)newG;

    output[i + 2] =
        (uchar)newB;
}
)CLC";

// Analyse brightness, contrast and RGB characteristics using OpenMP
ImageAnalysis analyseImage(const std::vector<unsigned char>& pixels)
{
    ImageAnalysis analysis{};

    if (pixels.empty())
    {
        analysis.averageBrightness = 0.0;
        analysis.contrast = 0.0;
        analysis.averageRed = 0.0;
        analysis.averageGreen = 0.0;
        analysis.averageBlue = 0.0;
        analysis.mode = CINEMATIC;
        return analysis;
    }

    long long pixelCount =
        static_cast<long long>(pixels.size() / 3);

    long long brightnessSum = 0;
    long long redSum = 0;
    long long greenSum = 0;
    long long blueSum = 0;

    // OpenMP threads analyse different pixels in parallel
    #pragma omp parallel for reduction(+:brightnessSum,redSum,greenSum,blueSum)
    for (long long p = 0; p < pixelCount; ++p)
    {
        long long i = p * 3;

        int red =
            static_cast<int>(pixels[i]);

        int green =
            static_cast<int>(pixels[i + 1]);

        int blue =
            static_cast<int>(pixels[i + 2]);

        int brightness =
            (red + green + blue) / 3;

        brightnessSum += brightness;
        redSum += red;
        greenSum += green;
        blueSum += blue;
    }

    analysis.averageBrightness =
        static_cast<double>(brightnessSum) / pixelCount;

    analysis.averageRed =
        static_cast<double>(redSum) / pixelCount;

    analysis.averageGreen =
        static_cast<double>(greenSum) / pixelCount;

    analysis.averageBlue =
        static_cast<double>(blueSum) / pixelCount;

    double squaredDifferenceSum = 0.0;

    // Parallel calculation of image contrast
    #pragma omp parallel for reduction(+:squaredDifferenceSum)
    for (long long p = 0; p < pixelCount; ++p)
    {
        long long i = p * 3;

        double brightness =
            (static_cast<double>(pixels[i]) +
             static_cast<double>(pixels[i + 1]) +
             static_cast<double>(pixels[i + 2])) / 3.0;

        double difference =
            brightness - analysis.averageBrightness;

        squaredDifferenceSum +=
            difference * difference;
    }

    analysis.contrast =
        std::sqrt(
            squaredDifferenceSum / pixelCount);

    // Thresholds used by the adaptive decision system
    const double colourDominanceThreshold = 8.0;
    const double lowContrastThreshold = 45.0;

    bool greenDominant =
        analysis.averageGreen >
            analysis.averageRed +
            colourDominanceThreshold &&
        analysis.averageGreen >
            analysis.averageBlue +
            colourDominanceThreshold;

    bool blueDominant =
        analysis.averageBlue >
            analysis.averageRed +
            colourDominanceThreshold &&
        analysis.averageBlue >
            analysis.averageGreen +
            colourDominanceThreshold;

    // Select the colour grade from measured image characteristics
    if (greenDominant)
    {
        analysis.mode =
            VIBRANT_NATURE;
    }
    else if (blueDominant)
    {
        analysis.mode =
            WARM_BALANCE;
    }
    else if (analysis.contrast <
             lowContrastThreshold)
    {
        analysis.mode =
            VIBRANT;
    }
    else
    {
        analysis.mode =
            CINEMATIC;
    }

    return analysis;
}

// Convert processing mode into readable text
std::string getModeName(ProcessingMode mode)
{
    switch (mode)
    {
        case VIBRANT_NATURE:
            return "VIBRANT NATURE";

        case WARM_BALANCE:
            return "WARM BALANCE";

        case VIBRANT:
            return "VIBRANT";

        case CINEMATIC:
            return "CINEMATIC";

        default:
            return "UNKNOWN";
    }
}

// Explain why the adaptive system selected the mode
std::string getClassification(ProcessingMode mode)
{
    switch (mode)
    {
        case VIBRANT_NATURE:
            return "Green-dominant / nature scene";

        case WARM_BALANCE:
            return "Blue-dominant / cool scene";

        case VIBRANT:
            return "Low-contrast / flat scene";

        case CINEMATIC:
            return "Balanced scene";

        default:
            return "Unknown";
    }
}

// Process one MPI image chunk using OpenCL
bool processWithOpenCL(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    ProcessingMode mode)
{
    if (input.empty())
        return true;

    cl_int error;
    cl_platform_id platform;

    error = clGetPlatformIDs(
        1,
        &platform,
        nullptr);

    if (error != CL_SUCCESS)
    {
        std::cerr
            << "OpenCL platform not found.\n";

        return false;
    }

    // Try GPU first and use CPU if no GPU is available
    cl_device_id device;

    error = clGetDeviceIDs(
        platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &device,
        nullptr);

    if (error != CL_SUCCESS)
    {
        error = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_CPU,
            1,
            &device,
            nullptr);
    }

    if (error != CL_SUCCESS)
    {
        std::cerr
            << "OpenCL device not found.\n";

        return false;
    }

    cl_context context =
        clCreateContext(
            nullptr,
            1,
            &device,
            nullptr,
            nullptr,
            &error);

    if (error != CL_SUCCESS)
        return false;

    cl_command_queue queue =
        clCreateCommandQueue(
            context,
            device,
            0,
            &error);

    if (error != CL_SUCCESS)
    {
        clReleaseContext(context);
        return false;
    }

    // Build the embedded OpenCL kernel
    cl_program program =
        clCreateProgramWithSource(
            context,
            1,
            &kernelSource,
            nullptr,
            &error);

    if (error != CL_SUCCESS)
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    error = clBuildProgram(
        program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr);

    if (error != CL_SUCCESS)
    {
        size_t logSize = 0;

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &logSize);

        std::vector<char>
            log(logSize);

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            logSize,
            log.data(),
            nullptr);

        std::cerr
            << "OpenCL build error:\n"
            << log.data()
            << "\n";

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return false;
    }

    cl_kernel kernel =
        clCreateKernel(
            program,
            "processImage",
            &error);

    if (error != CL_SUCCESS)
    {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    size_t dataSize =
        input.size() *
        sizeof(unsigned char);

    cl_mem inputBuffer =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY |
                CL_MEM_COPY_HOST_PTR,
            dataSize,
            const_cast<unsigned char*>(
                input.data()),
            &error);

    if (error != CL_SUCCESS)
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_mem outputBuffer =
        clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            dataSize,
            nullptr,
            &error);

    if (error != CL_SUCCESS)
    {
        clReleaseMemObject(inputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    int totalPixels =
        static_cast<int>(
            input.size() / 3);

    int modeValue =
        static_cast<int>(mode);

    error = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &inputBuffer);

    error |= clSetKernelArg(
        kernel,
        1,
        sizeof(cl_mem),
        &outputBuffer);

    error |= clSetKernelArg(
        kernel,
        2,
        sizeof(int),
        &totalPixels);

    error |= clSetKernelArg(
        kernel,
        3,
        sizeof(int),
        &modeValue);

    if (error != CL_SUCCESS)
    {
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    // One OpenCL work-item processes one complete RGB pixel
    size_t globalSize =
        static_cast<size_t>(
            totalPixels);

    error = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0,
        nullptr,
        nullptr);

    if (error != CL_SUCCESS)
    {
        std::cerr
            << "Failed to execute OpenCL kernel.\n";

        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return false;
    }

    clFinish(queue);

    output.resize(
        input.size());

    error = clEnqueueReadBuffer(
        queue,
        outputBuffer,
        CL_TRUE,
        0,
        dataSize,
        output.data(),
        0,
        nullptr,
        nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return error == CL_SUCCESS;
}

int main(int argc, char* argv[])
{
    MPI_Init(
        &argc,
        &argv);

    int rank;
    int processCount;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank);

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &processCount);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cout
                << "Usage: adaptive_image_processing <image_path>\n";
        }

        MPI_Finalize();
        return 1;
    }

    int width = 0;
    int height = 0;
    const int channels = 3;

    std::vector<unsigned char>
        fullImage;

    ImageAnalysis analysis{};

    double startTime =
        MPI_Wtime();

    // Rank 0 loads and analyses the complete image
    if (rank == 0)
    {
        int originalChannels;

        unsigned char* imageData =
            stbi_load(
                argv[1],
                &width,
                &height,
                &originalChannels,
                channels);

        if (imageData == nullptr)
        {
            std::cerr
                << "Error: Could not load input image.\n";

            width = 0;
        }
        else
        {
            int totalValues =
                width *
                height *
                channels;

            fullImage.assign(
                imageData,
                imageData + totalValues);

            stbi_image_free(
                imageData);

            double analysisStart =
                omp_get_wtime();

            analysis =
                analyseImage(
                    fullImage);

            double analysisEnd =
                omp_get_wtime();

            std::cout
                << "\nAdaptive Colour Grading System\n";

            std::cout
                << "--------------------------------\n";

            std::cout
                << "Image size: "
                << width
                << " x "
                << height
                << "\n";

            std::cout
                << "MPI processes: "
                << processCount
                << "\n";

            std::cout
                << "OpenMP threads: "
                << omp_get_max_threads()
                << "\n";

            std::cout
                << "Average brightness: "
                << analysis.averageBrightness
                << "\n";

            std::cout
                << "Contrast: "
                << analysis.contrast
                << "\n";

            std::cout
                << "Average red: "
                << analysis.averageRed
                << "\n";

            std::cout
                << "Average green: "
                << analysis.averageGreen
                << "\n";

            std::cout
                << "Average blue: "
                << analysis.averageBlue
                << "\n";

            std::cout
                << "Image classification: "
                << getClassification(
                    analysis.mode)
                << "\n";

            std::cout
                << "Selected colour grade: "
                << getModeName(
                    analysis.mode)
                << "\n";

            std::cout
                << "OpenMP analysis time: "
                << (analysisEnd -
                    analysisStart)
                << " seconds\n\n";
        }
    }

    // Tell all MPI ranks whether image loading succeeded
    MPI_Bcast(
        &width,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    if (width == 0)
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(
        &height,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    int modeValue = 0;

    if (rank == 0)
    {
        modeValue =
            static_cast<int>(
                analysis.mode);
    }

    // Tell every MPI rank which colour grade to apply
    MPI_Bcast(
        &modeValue,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    int totalPixels =
        width * height;

    // Calculate the number of complete pixels assigned to each MPI rank
    std::vector<int>
        sendCounts(processCount);

    std::vector<int>
        displacements(processCount);

    int basePixels =
        totalPixels /
        processCount;

    int remainingPixels =
        totalPixels %
        processCount;

    int displacement = 0;

    for (int i = 0;
         i < processCount;
         ++i)
    {
        int pixelsForRank =
            basePixels +
            (i < remainingPixels
                ? 1
                : 0);

        sendCounts[i] =
            pixelsForRank *
            channels;

        displacements[i] =
            displacement;

        displacement +=
            sendCounts[i];
    }

    int localSize =
        sendCounts[rank];

    std::vector<unsigned char>
        localInput(localSize);

    // MPI distributes image chunks between processes
    MPI_Scatterv(
        rank == 0
            ? fullImage.data()
            : nullptr,
        sendCounts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        localInput.data(),
        localSize,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    std::vector<unsigned char>
        localOutput;

    // Every MPI rank processes its chunk using OpenCL
    bool success =
        processWithOpenCL(
            localInput,
            localOutput,
            static_cast<ProcessingMode>(
                modeValue));

    if (!success)
    {
        std::cerr
            << "Rank "
            << rank
            << ": OpenCL processing failed.\n";

        MPI_Abort(
            MPI_COMM_WORLD,
            1);

        return 1;
    }

    std::vector<unsigned char>
        processedImage;

    if (rank == 0)
    {
        processedImage.resize(
            width *
            height *
            channels);
    }

    // Gather all processed chunks back onto Rank 0
    MPI_Gatherv(
        localOutput.data(),
        localSize,
        MPI_UNSIGNED_CHAR,
        rank == 0
            ? processedImage.data()
            : nullptr,
        sendCounts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::string outputPath =
            "output_images/processed.png";

        int saved =
            stbi_write_png(
                outputPath.c_str(),
                width,
                height,
                channels,
                processedImage.data(),
                width * channels);

        double endTime =
            MPI_Wtime();

        if (saved)
        {
            std::cout
                << "Output saved to: "
                << outputPath
                << "\n";
        }
        else
        {
            std::cerr
                << "Error: Could not save output image.\n";
        }

        std::cout
            << "Total execution time: "
            << (endTime - startTime)
            << " seconds\n";
    }

    MPI_Finalize();

    return 0;
}