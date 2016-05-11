#include "FAST/Algorithms/UsReconstruction/PnnNoHf.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Data/Image.hpp"
using namespace fast;

void PnnNoHf::setOutputType(DataType type){
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

PnnNoHf::PnnNoHf(){
    createInputPort<Image>(0);
    createOutputPort<Image>(0, OUTPUT_DEPENDS_ON_INPUT, 0);
    //create openCL prog here
    //--//createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter2D.cl", "2D");
    //--// store different compiled for settings (dimension/variables...)
    mIsModified = true; // needed?
    mOutputTypeSet = false;

    //volume;
    dv = 1;
    Rmax = 3; //2?
    volumeInitialized = false;
    firstFrameNotSet = true;
    frameList = {};
    //frameList.capacity = 1000;
}

PnnNoHf::~PnnNoHf(){
    //delete something
}

//void PnnNoHf::recompileOpenCLCode(Image::pointer input) {

Vector3f getImagePlaneNormal(Image::pointer frame){
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(frame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    return imagePlaneNormal;
}

// CPU algoritme
template <class T>
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output){
    ImageAccess::pointer volAccess = VoxelsValNWeight->getImageAccess(ACCESS_READ_WRITE);
    Image::pointer lastFrame = None;
    while (!frameList.empty()){
        Image::pointer frame = frameList.back();
        frameList.pop_back();
        //zeroPoints
        // # Finn dominerende rettning #
        Vector3f imagePlaneNormal = getImagePlaneNormal(frame);
        float domVal = imagePlaneNormal(0); int domDir = 0
        if (imagePlaneNormal(1) > domDir) domVal = imagePlaneNormal(1); domDir = 1;
        if (imagePlaneNormal(2) > domDir) domVal = imagePlaneNormal(2); domDir = 2;
        // TODO adjust for or compare to output volume direction

        // # Go through output volume #
        ImageAccess::pointer frameAccess = frame->getImageAccess(ACCESS_READ);
        if (!frameList.empty()){ Image::pointer next = frameList.back(); }


        for (int x = 0; x < frame.getWidth(); x++){
            for (int y = 0; y < frame.getHeight(); y++){
                //Find thickness according to last and next frame along the imagePlaneNormal
                float maxNeighDist = max(d1, d2);
                float df = min(max(maxNeighDist, dv), Rmax);
                float dfz = df / domVal;
            }
        }
    }
}
/*
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output, float * mask, unsigned char maskSize) {
    // TODO: this method currently only processes the first component
    unsigned int nrOfComponents = input->getNrOfComponents();
    ImageAccess::pointer inputAccess = input->getImageAccess(ACCESS_READ);
    ImageAccess::pointer outputAccess = output->getImageAccess(ACCESS_READ_WRITE);

    T * inputData = (T*)inputAccess->get();
    T * outputData = (T*)outputAccess->get();

    const unsigned char halfSize = (maskSize - 1) / 2;
    unsigned int width = input->getWidth();
    unsigned int height = input->getHeight();
    if (input->getDimensions() == 3) {
        unsigned int depth = input->getDepth();
        for (unsigned int z = 0; z < depth; z++) {
            for (unsigned int y = 0; y < height; y++) {
                for (unsigned int x = 0; x < width; x++) {

                    if (x < halfSize || x >= width - halfSize ||
                        y < halfSize || y >= height - halfSize ||
                        z < halfSize || z >= depth - halfSize) {
                        // on border only copy values
                        outputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height] = inputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height];
                        continue;
                    }

                    double sum = 0.0;
                    for (int c = -halfSize; c <= halfSize; c++) {
                        for (int b = -halfSize; b <= halfSize; b++) {
                            for (int a = -halfSize; a <= halfSize; a++) {
                                sum += mask[a + halfSize + (b + halfSize)*maskSize + (c + halfSize)*maskSize*maskSize] *
                                    inputData[(x + a)*nrOfComponents + (y + b)*nrOfComponents*width + (z + c)*nrOfComponents*width*height];
                            }
                        }
                    }
                    outputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height] = (T)sum;
                }
            }
        }
    }
    else {
        for (unsigned int y = halfSize; y < height - halfSize; y++) {
            for (unsigned int x = halfSize; x < width - halfSize; x++) {

                if (x < halfSize || x >= width - halfSize ||
                    y < halfSize || y >= height - halfSize) {
                    // on border only copy values
                    outputData[x*nrOfComponents + y*nrOfComponents*width] = inputData[x*nrOfComponents + y*nrOfComponents*width];
                    continue;
                }

                double sum = 0.0;
                for (int b = -halfSize; b <= halfSize; b++) {
                    for (int a = -halfSize; a <= halfSize; a++) {
                        sum += mask[a + halfSize + (b + halfSize)*maskSize] *
                            inputData[(x + a)*nrOfComponents + (y + b)*nrOfComponents*width];
                    }
                }
                outputData[x*nrOfComponents + y*nrOfComponents*width] = (T)sum;
            }
        }
    }
}*/

void PnnNoHf::execute() {
    //Image::pointer input = getStaticInputData<Image>(0);

    Image::pointer frame = getStaticInputData<Image>(0);
    //Image::pointer output = getStaticOutputData<Image>(0);
    //output->createFromImage(frame);
    //setStaticOutputData<Image>(0, frame);
    //return;
    if (firstFrameNotSet){
        firstFrame = frame;
        //frameList.push_back(frame);
        firstFrameNotSet = false;

        /*DynamicData::pointer dynamicData = getInputData(0);
        dynamicData->registerConsumer(this);

        firstFrame = dynamicData->getNextFrame(this);
        frameList.push_back(firstFrame);
        output->createFromImage(firstFrame);

        while (!dynamicData->hasReachedEnd()){
            Image::pointer nextFrame = dynamicData->getNextFrame(this);
            frameList.push_back(nextFrame);
        }

        if (!volumeInitialized){
            std::cout << "INITIALIZING volume" << std::endl;
            //Init cube with all corners
            initVolumeCube(firstFrame);
            volumeInitialized = true;
            //Definer dv (oppløsning)
            dv = 1;
        }
        switch (firstFrame->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(firstFrame, output));
        }*/
            
        //output = firstFrame;
        /*IMAGE constructors
        void create(VectorXui size, DataType type, uint nrOfComponents);
        void create(uint width, uint height, uint depth, DataType type, uint nrOfComponents);
        void create(VectorXui size, DataType type, uint nrOfComponents, ExecutionDevice::pointer device, const void * data);
        void create(uint width, uint height, uint depth, DataType type, uint nrOfComponents, ExecutionDevice::pointer device, const void * data);
        */
        //frame->getNrOfComponents;
        //frame->getDataType();
        //Image::pointer output = getStaticOutputData<Image>();
        //output->create(input->getSize(), TYPE_FLOAT, input->getNrOfComponents());
        
        
        output = getStaticOutputData<Image>(0);
        DataType type = DataType::TYPE_INT8; //frame->getDataType();
        uint size = 32;
        int initVal = 1;
        output->create(size, size, size, type, 1);// create(500, 500, 500, frame->getDataType(), 2);
        ImageAccess::pointer imgAccess = output->getImageAccess(accessType::ACCESS_READ_WRITE);
        ImageAccess::pointer inpAccess = frame->getImageAccess(accessType::ACCESS_READ);
        for (int x = 0; x < size; x++){
            for (int y = 0; y < size; y++){
                int thisVal = inpAccess->getScalar((x, y), 0);
                for (int z = 0; z < size; z++){
                    //imgAccess->setScalar((x, y, z), initVal, 0); //Channel 1 - Value
                    //imgAccess->setScalar((x, y, z), initVal, 1); //Channel 2 - Weight
                    imgAccess->setScalar((x, y, z), thisVal, 0);

                    //imgAccess->setVector(Eigen::Vector3i(x, y, z), Eigen::Vector2i(0, 0)); // Eventuelt Vector2f etc
                }
            }
        }
        imgAccess->release();
        inpAccess->release();
        std::cout << "MAX intensity" << output->calculateMaximumIntensity() << std::endl;
        
        //setStaticOutputData<Image>(0, output);

        //output->setSpacing(TODO);
    }
    // Lagre frame i PO, f.eks. i en std::vector
    
    setStaticOutputData<Image>(0, output);
     
    frameList.push_back(frame);
    
    //Image::pointer output = getStaticOutputData<Image>();
    //output->create(256, 256, 256, frame->getDataType(), 2);
    // Sjekk om vi har nådd slutten
    DynamicData::pointer dynamicImage = getInputData(0);
    //dynamicImage->
    if (dynamicImage->hasReachedEnd()) {
        //Image::pointer output = getStaticOutputData<Image>(0);
        // Do reconstruction
        
        //output->setDimension(3);

        if (!volumeInitialized){
            std::cout << "INITIALIZING volume" << std::endl;
            //Init cube with all corners
            initVolumeCube(firstFrame);
            volumeInitialized = true;
            //Definer dv (oppløsning)
            dv = 1;
        }
        switch (frame->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(frame, output));
        }
    }
    else{// if (dynamicImage->getSize() == 0){
        std::cout << "DynImg size" << dynamicImage->getSize() << std::endl;
    }
        //getInputData(0);//getStaticInputData<Image>(0);
    /*if (input->getDimension() != 2){
        throw Exception("The algorithm only handles 2D image input");
    }*/
    //if (dynamicImage->)
    
    return;
    /*if (device->isHost()){
        switch (input->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
        }
    }
    else{
        switch (input->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
        }
    }*/

    /*
    char maskSize = mMaskSize;
    if (maskSize <= 0) // If mask size is not set calculate it instead
        maskSize = ceil(2 * mStdDev) * 2 + 1;

    if (maskSize > 19)
        maskSize = 19;

    // Initialize output image
    ExecutionDevice::pointer device = getMainDevice();
    if (mOutputTypeSet) {
        output->create(input->getSize(), mOutputType, input->getNrOfComponents());
        output->setSpacing(input->getSpacing());
    }
    else {
        output->createFromImage(input);
    }
    mOutputType = output->getDataType();
    SceneGraph::setParentNode(output, input);


    if (device->isHost()) {
        createMask(input, maskSize, false);
        switch (input->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
        }
    }
    else {
        OpenCLDevice::pointer clDevice = device;

        recompileOpenCLCode(input);

        cl::NDRange globalSize;

        OpenCLImageAccess::pointer inputAccess = input->getOpenCLImageAccess(ACCESS_READ, device);
        if (input->getDimensions() == 2) {
            createMask(input, maskSize, false);
            mKernel.setArg(1, mCLMask);
            mKernel.setArg(3, maskSize);
            globalSize = cl::NDRange(input->getWidth(), input->getHeight());

            OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
            mKernel.setArg(0, *inputAccess->get2DImage());
            mKernel.setArg(2, *outputAccess->get2DImage());
            clDevice->getCommandQueue().enqueueNDRangeKernel(
                mKernel,
                cl::NullRange,
                globalSize,
                cl::NullRange
                );
        }
        else {
            // Create an auxilliary image
            Image::pointer output2 = Image::New();
            output2->createFromImage(output);

            globalSize = cl::NDRange(input->getWidth(), input->getHeight(), input->getDepth());

            if (clDevice->isWritingTo3DTexturesSupported()) {
                createMask(input, maskSize, true);
                mKernel.setArg(1, mCLMask);
                mKernel.setArg(3, maskSize);
                OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
                OpenCLImageAccess::pointer outputAccess2 = output2->getOpenCLImageAccess(ACCESS_READ_WRITE, device);

                cl::Image3D* image2;
                cl::Image3D* image;
                image = outputAccess->get3DImage();
                image2 = outputAccess->get3DImage();
                for (uchar direction = 0; direction < input->getDimensions(); ++direction) {
                    if (direction == 0) {
                        mKernel.setArg(0, *inputAccess->get3DImage());
                        mKernel.setArg(2, *image);
                    }
                    else if (direction == 1) {
                        mKernel.setArg(0, *image);
                        mKernel.setArg(2, *image2);
                    }
                    else {
                        mKernel.setArg(0, *image2);
                        mKernel.setArg(2, *image);
                    }
                    mKernel.setArg(4, direction);
                    clDevice->getCommandQueue().enqueueNDRangeKernel(
                        mKernel,
                        cl::NullRange,
                        globalSize,
                        cl::NullRange
                        );
                }
            }
            else {
                createMask(input, maskSize, false);
                mKernel.setArg(1, mCLMask);
                mKernel.setArg(3, maskSize);
                OpenCLBufferAccess::pointer outputAccess = output->getOpenCLBufferAccess(ACCESS_READ_WRITE, device);
                mKernel.setArg(0, *inputAccess->get3DImage());
                mKernel.setArg(2, *outputAccess->get());
                clDevice->getCommandQueue().enqueueNDRangeKernel(
                    mKernel,
                    cl::NullRange,
                    globalSize,
                    cl::NullRange
                    );
            }


        }
    }*/
}

void PnnNoHf::initVolumeCube(Image::pointer rootFrame){
    /*
    bool dynData = input->isDynamicData();
    std::cout << "Is dynamic?" << dynData << std::endl;

    BoundingBox bb = input->getBoundingBox();
    std::cout << "Bounding box" << bb << std::endl;

    uint comps = input->getNrOfComponents();
    std::cout << "Components" << comps << std::endl;

    Vector3ui size = input->getSize();
    std::cout << "Size" << size << std::endl;
    //Vector3ui spacing = input->getSpacing();
    //std::cout << "Spacing" << spacing << std::endl;
    Streamer::pointer stream = input->getStreamer();
    std::cout << "Stream " << stream << std::endl;
    //uint nrOfFrames = stream->getNrOfFrames();
    */
    // Calculate image plane normal
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(rootFrame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    
    // Define zero point
    BoundingBox rootBB = rootFrame->getBoundingBox();
    uint width = rootFrame->getWidth();
    uint height = rootFrame->getHeight();
    uint components = rootFrame->getNrOfComponents();
    BoundingBox rootBBtrans = rootFrame->getTransformedBoundingBox();
    Eigen::Vector3f spacings = rootFrame->getSpacing();    

    Eigen::MatrixXf rootCorners = rootBB.getCorners();
    Eigen::MatrixXf rootBBtransCorners = rootBBtrans.getCorners();

    //Eigen::Vector3f cornerOne = rootCorners(0);
    std::vector<Eigen::Vector3f> points = {};
    for (int i = 0; i < 8; i++){
        Eigen::Vector3f corner(rootCorners(i, 0), rootCorners(i, 1), rootCorners(i, 2));
        Eigen::Vector3f cornerTrans(rootBBtransCorners(i, 0), rootBBtransCorners(i, 1), rootBBtransCorners(i, 2));
        points.push_back(cornerTrans);
    }

    // BIG TODO
        // Find transform from rootFrame to normalized space
        // Use this transform to transform all frames
    
    Vector3f min, max;
    //Vector3f centroid;
    //BoundingBox box = rootFrame->getBoundingBox();
    BoundingBox box = rootFrame->getTransformedBoundingBox();
    MatrixXf corners = box.getCorners();
    Vector3f corner = box.getCorners().row(0);
    min[0] = corner[0];
    max[0] = corner[0];
    min[1] = corner[1];
    max[1] = corner[1];
    min[2] = corner[2];
    max[2] = corner[2];
   
    
    for (int i = 0; i < frameList.size(); i++){
        BoundingBox box = frameList[i]->getTransformedBoundingBox();
        MatrixXf corners = box.getCorners();
        for (int j = 0; j < 8; j++) {
            for (uint k = 0; k < 3; k++) {
                float point = corners(j, k);
                if (point < min[k]) //corners(j, k) < min[k])
                    min[k] = point;
                if (point > max[k])
                    max[k] = point;
            }
        }
    }
    

    // Calculate directions relative in this one
    //  - x: horizontal; y: vertical in img; z: (0) in all of rootFrame

    // Init volume
    Vector3f size = max - min;
    zeroPoints = min;
    DataType type = DataType::TYPE_INT8; //frame->getDataType();
    int initVal = 0.0;
    VoxelsValNWeight->create(size, type, 2);
    ImageAccess::pointer volAccess = VoxelsValNWeight->getImageAccess(accessType::ACCESS_READ_WRITE);
    for (int x = 0; x < size[0]; x++){
        for (int y = 0; y < size[0]; y++){
            for (int z = 0; z < size[0]; z++){
                volAccess->setScalar((x, y, z), initVal, 0); //Channel 1 - Value
                volAccess->setScalar((x, y, z), initVal, 1); //Channel 2 - Weight

                //imgAccess->setVector(Eigen::Vector3i(x, y, z), Eigen::Vector2i(0, 0)); // Eventuelt Vector2f etc
            }
        }
    }
    volAccess->release();

    //calculate dv?
    int drrrrrrrr = 1;
}

void PnnNoHf::waitToFinish() {
    if (!getMainDevice()->isHost()) {
        OpenCLDevice::pointer device = getMainDevice();
        device->getCommandQueue().finish();
    }
}

// ########################
/*
void GaussianSmoothingFilter::setMaskSize(unsigned char maskSize) {
    if(maskSize <= 0)
        throw Exception("Mask size of GaussianSmoothingFilter can't be less than 0.");
    if(maskSize % 2 != 1)
        throw Exception("Mask size of GaussianSmoothingFilter must be odd.");

    mMaskSize = maskSize;
    mIsModified = true;
    mRecreateMask = true;
}

void GaussianSmoothingFilter::setOutputType(DataType type) {
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

void GaussianSmoothingFilter::setStandardDeviation(float stdDev) {
    if(stdDev <= 0)
        throw Exception("Standard deviation of GaussianSmoothingFilter can't be less than 0.");

    mStdDev = stdDev;
    mIsModified = true;
    mRecreateMask = true;
}

GaussianSmoothingFilter::GaussianSmoothingFilter() {
    createInputPort<Image>(0);
    createOutputPort<Image>(0, OUTPUT_DEPENDS_ON_INPUT, 0);
    createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter2D.cl", "2D");
    createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter3D.cl", "3D");
    mStdDev = 0.5f;
    mMaskSize = -1;
    mIsModified = true;
    mRecreateMask = true;
    mDimensionCLCodeCompiledFor = 0;
    mMask = NULL;
    mOutputTypeSet = false;
}

GaussianSmoothingFilter::~GaussianSmoothingFilter() {
    delete[] mMask;
}

// TODO have to set mRecreateMask to true if input change dimension
void GaussianSmoothingFilter::createMask(Image::pointer input, uchar maskSize, bool useSeperableFilter) {
    if(!mRecreateMask)
        return;

    unsigned char halfSize = (maskSize-1)/2;
    float sum = 0.0f;

    if(input->getDimensions() == 2) {
        mMask = new float[maskSize*maskSize];

        for(int x = -halfSize; x <= halfSize; x++) {
        for(int y = -halfSize; y <= halfSize; y++) {
            float value = exp(-(float)(x*x+y*y)/(2.0f*mStdDev*mStdDev));
            mMask[x+halfSize+(y+halfSize)*maskSize] = value;
            sum += value;
        }}

        for(int i = 0; i < maskSize*maskSize; ++i)
            mMask[i] /= sum;
    } else if(input->getDimensions() == 3) {
        // Use separable filtering for 3D
        if(useSeperableFilter) {
            mMask = new float[maskSize];

            for(int x = -halfSize; x <= halfSize; x++) {
                float value = exp(-(float)(x*x)/(2.0f*mStdDev*mStdDev));
                mMask[x+halfSize] = value;
                sum += value;
            }

            for(int i = 0; i < maskSize; ++i)
                mMask[i] /= sum;
        } else {
            mMask = new float[maskSize*maskSize*maskSize];

            for(int x = -halfSize; x <= halfSize; x++) {
            for(int y = -halfSize; y <= halfSize; y++) {
            for(int z = -halfSize; z <= halfSize; z++) {
                float value = exp(-(float)(x*x+y*y+z*z)/(2.0f*mStdDev*mStdDev));
                mMask[x+halfSize+(y+halfSize)*maskSize+(z+halfSize)*maskSize*maskSize] = value;
                sum += value;
            }}}

            for(int i = 0; i < maskSize*maskSize*maskSize; ++i)
                mMask[i] /= sum;
        }
    }

    ExecutionDevice::pointer device = getMainDevice();
    if(!device->isHost()) {
        OpenCLDevice::pointer clDevice = device;
        uint bufferSize;
        if(useSeperableFilter) {
            bufferSize = maskSize;
        } else {
            bufferSize = input->getDimensions() == 2 ? maskSize*maskSize : maskSize*maskSize*maskSize;
        }
        mCLMask = cl::Buffer(
                clDevice->getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float)*bufferSize,
                mMask
        );
    }

    mRecreateMask = false;
}

void GaussianSmoothingFilter::recompileOpenCLCode(Image::pointer input) {
    // Check if there is a need to recompile OpenCL code
    if(input->getDimensions() == mDimensionCLCodeCompiledFor &&
            input->getDataType() == mTypeCLCodeCompiledFor)
        return;

    OpenCLDevice::pointer device = getMainDevice();
    std::string buildOptions = "";
    if(!device->isWritingTo3DTexturesSupported()) {
        buildOptions = "-DTYPE=" + getCTypeAsString(mOutputType);
    }
    cl::Program program;
    if(input->getDimensions() == 2) {
        program = getOpenCLProgram(device, "2D", buildOptions);
    } else {
        program = getOpenCLProgram(device, "3D", buildOptions);
    }
    mKernel = cl::Kernel(program, "gaussianSmoothing");
    mDimensionCLCodeCompiledFor = input->getDimensions();
    mTypeCLCodeCompiledFor = input->getDataType();
}

template <class T>
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output, float * mask, unsigned char maskSize) {
    // TODO: this method currently only processes the first component
    unsigned int nrOfComponents = input->getNrOfComponents();
    ImageAccess::pointer inputAccess = input->getImageAccess(ACCESS_READ);
    ImageAccess::pointer outputAccess = output->getImageAccess(ACCESS_READ_WRITE);

    T * inputData = (T*)inputAccess->get();
    T * outputData = (T*)outputAccess->get();

    const unsigned char halfSize = (maskSize-1)/2;
    unsigned int width = input->getWidth();
    unsigned int height = input->getHeight();
    if(input->getDimensions() == 3) {
        unsigned int depth = input->getDepth();
        for(unsigned int z = 0; z < depth; z++) {
        for(unsigned int y = 0; y < height; y++) {
        for(unsigned int x = 0; x < width; x++) {

            if(x < halfSize || x >= width-halfSize ||
            y < halfSize || y >= height-halfSize ||
            z < halfSize || z >= depth-halfSize) {
                // on border only copy values
                outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = inputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height];
                continue;
            }

            double sum = 0.0;
            for(int c = -halfSize; c <= halfSize; c++) {
            for(int b = -halfSize; b <= halfSize; b++) {
            for(int a = -halfSize; a <= halfSize; a++) {
                sum += mask[a+halfSize+(b+halfSize)*maskSize+(c+halfSize)*maskSize*maskSize]*
                        inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width+(z+c)*nrOfComponents*width*height];
            }}}
            outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = (T)sum;
        }}}
    } else {
        for(unsigned int y = halfSize; y < height-halfSize; y++) {
        for(unsigned int x = halfSize; x < width-halfSize; x++) {

            if(x < halfSize || x >= width-halfSize ||
            y < halfSize || y >= height-halfSize) {
                // on border only copy values
                outputData[x*nrOfComponents+y*nrOfComponents*width] = inputData[x*nrOfComponents+y*nrOfComponents*width];
                continue;
            }

            double sum = 0.0;
            for(int b = -halfSize; b <= halfSize; b++) {
            for(int a = -halfSize; a <= halfSize; a++) {
                sum += mask[a+halfSize+(b+halfSize)*maskSize]*
                        inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width];
            }}
            outputData[x*nrOfComponents+y*nrOfComponents*width] = (T)sum;
        }}
    }
    
}*/


