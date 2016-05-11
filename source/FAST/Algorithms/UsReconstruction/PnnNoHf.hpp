#ifndef PNN_NOHF_HPP_
#define PNN_NOHF_HPP_

#include "FAST/ProcessObject.hpp"
#include "FAST/ExecutionDevice.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {

class PnnNoHf : public ProcessObject {
    FAST_OBJECT(PnnNoHf)
    public:
        //void setMaskSize(unsigned char maskSize);
        //void setStandardDeviation(float stdDev);
        void setOutputType(DataType type);
        ~PnnNoHf();
    private:
        PnnNoHf();
        void execute();
        void waitToFinish();
        void initVolumeCube(Image::pointer input);
        void executeAlgorithmOnHost();
        //void createMask(Image::pointer input, uchar maskSize, bool useSeperableFilter);
        //void recompileOpenCLCode(Image::pointer input);

        //char mMaskSize;
        //float mStdDev;
        float dv; //resolution?
        float Rmax;
        bool volumeInitialized;
        Image::pointer firstFrame;
        Image::pointer output;
        bool firstFrameNotSet;
        
        std::vector<Image::pointer> frameList;
        Image::pointer VoxelsValNWeight;
        float * VoxelValues;
        float * VoxelWeights;
        Vector3f zeroPoints;
        Vector3f volumeSize;


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

#endif /* PNN_NOHF_HPP_ */
