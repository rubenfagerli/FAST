#include "FAST/Algorithms/UsReconstruction/PnnTestVolume.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Data/Image.hpp"
using namespace fast;

void PnnTestVolume::setOutputType(DataType type){
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

PnnTestVolume::PnnTestVolume(){
    createInputPort<Image>(0);
    createOutputPort<Image>(0, OUTPUT_DEPENDS_ON_INPUT, 0);
    //create openCL prog here
    //--//createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter2D.cl", "2D");
    //--// store different compiled for settings (dimension/variables...)
    mIsModified = true; // needed?
    mOutputTypeSet = false;

    //volume;
    volumeCalculated = false;
    volumeInitialized = false;
    firstFrameNotSet = true;
    reachedEndOfStream = false;
    frameList = {};
    iterartorCounter = 0;
}

PnnTestVolume::~PnnTestVolume(){
    //delete something
}

void PnnTestVolume::makeVolumeFromFrame(Image::pointer frame){
    volAccess = output->getImageAccess(accessType::ACCESS_READ_WRITE);
    ImageAccess::pointer frameAccess = frame->getImageAccess(accessType::ACCESS_READ);
    Vector3ui volumeSize = output->getSize();
    Vector3ui frameSize = frame->getSize();
    for (int x = 0; x < volumeSize(0); x++){
        for (int y = 0; y < volumeSize(1); y++){
            if (x < frameSize(0) && y < frameSize(1)){
                float pixelValue = frameAccess->getScalar(Vector2i(x, y), 0);
                for (int z = 0; z < volumeSize(2); z++){
                    volAccess->setScalar(Vector3i(x, y, z), pixelValue, 0);
                }
            }
        }
    }
    volAccess.release();
    frameAccess.release();
}

void PnnTestVolume::execute(){
    Image::pointer frame = getStaticInputData<Image>(0);
    frameList.push_back(frame);
    if (firstFrameNotSet){
        firstFrame = frame;
        firstFrameNotSet = false;
        //Init volume
        output = getStaticOutputData<Image>(0);
        DataType type = DataType::TYPE_FLOAT;
        uint size = 128;
        float initVal = 1.0;
        output->create(size, size, size, type, 1);// create(500, 500, 500, frame->getDataType(), 2);
        volAccess = output->getImageAccess(accessType::ACCESS_READ_WRITE);
        for (int x = 0; x < size; x++){
            for (int y = 0; y < size; y++){
                for (int z = 0; z < size; z++){
                    volAccess->setScalar((x, y, z), 0.0, 0);
                }
            }
        }
        volAccess->release();
    }
    //Make volume
    makeVolumeFromFrame(frame);
    setStaticOutputData<Image>(0, output);
    /*
    if (!reachedEndOfStream){
        std::cout << "Iteration #:" << iterartorCounter++ << std::endl;
        Image::pointer frame = getStaticInputData<Image>(0);
        frameList.push_back(frame);
        if (firstFrameNotSet){
            firstFrame = frame;
            firstFrameNotSet = false;
        }
        // Sjekk om vi har nådd slutten
        DynamicData::pointer dynamicImage = getInputData(0);
        if (dynamicImage->hasReachedEnd()) {
            reachedEndOfStream = true;
        }
        setStaticOutputData<Image>(0, frame);
    }
    // When we have reached the end of stream we do just from here on
    if (reachedEndOfStream) {
        std::cout << "END Iteration #:" << iterartorCounter++ << std::endl;
        if (!volumeCalculated){
            if (!volumeInitialized){
                std::cout << "Nr of frames in frameList:" << frameList.size() << std::endl;
                std::cout << "INITIALIZING volume" << std::endl;
                //Init cube with all corners
                initVolumeCube(firstFrame);
                volumeInitialized = true;
                //Definer dv (oppløsning)
                dv = 1;
                outputImg = firstFrame;
            }
            //if use GPU else :
            executeAlgorithmOnHost();
        }
        setStaticOutputData<Image>(0, outputImg);
    }
    */
}

void PnnTestVolume::waitToFinish() {
    if (!getMainDevice()->isHost()) {
        OpenCLDevice::pointer device = getMainDevice();
        device->getCommandQueue().finish();
    }
}
