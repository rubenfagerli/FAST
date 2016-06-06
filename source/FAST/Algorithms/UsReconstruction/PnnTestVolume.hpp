#ifndef PNN_TEST_HPP_
#define PNN_TEST_HPP_

#include "FAST/ProcessObject.hpp"
#include "FAST/ExecutionDevice.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {

class PnnTestVolume : public ProcessObject {
    FAST_OBJECT(PnnTestVolume)
    public:
        void setOutputType(DataType type);
        ~PnnTestVolume();
    private:
        PnnTestVolume();
        void execute();
        void waitToFinish();
        void executeAlgorithmOnHost();
        void makeVolumeFromFrame(Image::pointer frame);
        /*
        void initVolume(Image::pointer input);
        void initVolumeCube(Image::pointer input);//EXPIRED
        std::vector<Image::pointer> getImagesAround(Vector3f worldPoint);
        void filterOutImagesByDfDom(std::vector<Image::pointer> neighbourFrames, int domDir, Vector3f worldPoint, float dfDom);
        void accumulateValue(Vector3i pointVoxelPos, float addValue, int channel);//EXPIRED
        void accumulateValuesInVolume(Vector3i volumePoint, float p, float w);
        Vector3f getFramePointPosition(Image::pointer frame, int x, int y);
        
        //void createMask(Image::pointer input, uchar maskSize, bool useSeperableFilter);
        //void recompileOpenCLCode(Image::pointer input);
        */

        //char mMaskSize;
        //float mStdDev;
        float dv; //resolution?
        float Rmax;
        bool volumeCalculated;
        bool volumeInitialized;
        Image::pointer firstFrame;
        Image::pointer output;
        Image::pointer outputImg;
        bool firstFrameNotSet;
        bool reachedEndOfStream;
        
        std::vector<Image::pointer> frameList;
        Image::pointer VoxelsValNWeight;
        Image::pointer AccumulationVolume;
        float * VoxelValues;
        float * VoxelWeights;
        Vector3f zeroPoints;
        Vector3i volumeSize;

        //Lists to store calculations done for different frames
        std::vector<Vector3f> frameMinList;
        std::vector<Vector3f> frameMaxList;
        std::vector<Vector3f> frameBaseCornerList;
        std::vector<Vector3f> framePlaneNormalList;

        ImageAccess::pointer volAccess;

        int iterartorCounter;


        cl::Buffer mCLMask;
        //float * mMask;
        //bool mRecreateMask;

        cl::Kernel mKernel;
        unsigned char mDimensionCLCodeCompiledFor;
        DataType mTypeCLCodeCompiledFor;
        DataType mOutputType;
        bool mOutputTypeSet;

};

} // end namespace fast

#endif /* PNN_TEST_HPP_ */
