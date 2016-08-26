//__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

float  readAndNormalize(__global unsigned int* volume, int loc, float defValue){
    int locReadP = loc * 2;
    int locReadW = loc * 2 + 1;
    unsigned int p = volume[locReadP];
    unsigned int w = volume[locReadW];
    float value = defValue;
    if (w != 0)
        value = (float)p / (float)w;
    return value;
}

/*
void setLocalValue(__local float* localStorage, int x, int y, int z, float valA){
int locA = (x + y*LSIZE_MEM_X + z*LSIZE_MEM_XtY);
localStorage[locA] = valA; //?
}
*/


__kernel void normalizeHoleFillVolume(
    __write_only image3d_t outputVolume,
    __global unsigned int * volume
    ){
    //,__local float * localStorage
    //Depends upon defined
    // VOL_SIZE_X/Y/Z/XtY
    // LSIZE_MEM_X/Y/Z/XtY = LocalMemory Size
    // LSIZE_X/Y/Z = workSize in local by direction
    const int xG = get_global_id(0);
    const int yG = get_global_id(1);
    const int zG = get_global_id(2);
    //int3 pos = (int3)(xG, yG, zG);
    int4 pos = (int4)(xG, yG, zG, 0);

    //const int x = get_local_id(0);
    // const int y = get_local_id(1);
    // const int z = get_local_id(2);

    //LOCAL OFFSET?
    int locGlobal = (xG + yG*VOL_SIZE_X + zG*VOL_SIZE_XtY); //Component later
    //float voxelValue = 0.0f; // readAndNormalize(volume, locGlobal, 0.0f);
    float voxelValue = readAndNormalize(volume, locGlobal, -1.0f);
    /*if (false){
    //READ A and normalize
    int AaddX = 0; int AaddY = 0; int AaddZ = 0;
    {
    int locGlobalA = ((xG + AaddX) + (yG + AaddY)*VOL_SIZE_X + (zG + AaddZ)*VOL_SIZE_XtY); //Component later
    float valA = readAndNormalize(volume, locGlobalA, -1.0f);
    setLocalValue(localStorage, (x + AaddX), (y + AaddY), (z + AaddZ), valA);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //READ B and normalize
    int BaddX = 0; int BaddY = LSIZE_Y; int BaddZ = 0;
    if ((y + BaddY) < LSIZE_MEM_Y && (yG + BaddY) < VOL_SIZE_Y){
    int locGlobalB = ((xG + BaddX) + (yG + BaddY)*VOL_SIZE_X + (zG + BaddZ)*VOL_SIZE_XtY); //Component later
    float valB = readAndNormalize(volume, locGlobalB, -1.0f);
    setLocalValue(localStorage, (x + BaddX), (y + BaddY), (z + BaddZ), valB);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //READ C and normalize
    int CaddX = LSIZE_X; int CaddY = 0; int CaddZ = 0;
    if ((x + CaddX) < LSIZE_MEM_X && (xG + CaddX) < VOL_SIZE_X){
    int locGlobalC = ((xG + CaddX) + (yG + CaddY)*VOL_SIZE_X + (zG + CaddZ)*VOL_SIZE_XtY); //Component later
    float valC = readAndNormalize(volume, locGlobalC, -1.0f);
    setLocalValue(localStorage, (x + CaddX), (y + CaddY), (z + CaddZ), valC);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //READ D and normalize
    int DaddX = LSIZE_X; int DaddY = LSIZE_Y; int DaddZ = 0;
    if ((x + DaddX) < LSIZE_MEM_X && (xG + DaddX) < VOL_SIZE_X
    && (y + DaddY) < LSIZE_MEM_Y && (yG + DaddY) < VOL_SIZE_Y){
    int locGlobalD = ((xG + DaddX) + (yG + DaddY)*VOL_SIZE_X + (zG + DaddZ)*VOL_SIZE_XtY); //Component later
    float valD = readAndNormalize(volume, locGlobalD, -1.0f);
    setLocalValue(localStorage, (x + DaddX), (y + DaddY), (z + DaddZ), valD);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int addZ = LSIZE_Z;
    if ((z + addZ) < LSIZE_MEM_Z && (zG + addZ) < VOL_SIZE_Z){
    //READ E and normalize
    int EaddX = 0; int EaddY = 0;
    {
    int locGlobalE = ((xG + EaddX) + (yG + EaddY)*VOL_SIZE_X + (zG + addZ)*VOL_SIZE_XtY); //Component later
    float valE = readAndNormalize(volume, locGlobalE, -1.0f);
    setLocalValue(localStorage, (x + EaddX), (y + EaddY), (z + addZ), valE);
    }
    //MEMBLOCK?
    //READ G and normalize
    int GaddX = 0; int GaddY = LSIZE_Y;
    if ((y + GaddY) < LSIZE_MEM_Y && (yG + GaddY) < VOL_SIZE_Y){
    int locGlobalG = ((xG + GaddX) + (yG + GaddY)*VOL_SIZE_X + (zG + addZ)*VOL_SIZE_XtY); //Component later
    float valG = readAndNormalize(volume, locGlobalG, -1.0f);
    setLocalValue(localStorage, (x + GaddX), (y + GaddY), (z + addZ), valG);
    }
    //MEMBLOCK?
    //READ F and normalize
    int FaddX = LSIZE_X; int FaddY = 0;
    if ((x + FaddX) < LSIZE_MEM_X && (xG + FaddX) < VOL_SIZE_X){
    int locGlobalF = ((xG + FaddX) + (yG + FaddY)*VOL_SIZE_X + (zG + addZ)*VOL_SIZE_XtY); //Component later
    float valF = readAndNormalize(volume, locGlobalF, -1.0f);
    setLocalValue(localStorage, (x + FaddX), (y + FaddY), (z + addZ), valF);
    }
    //MEMBLOCK?
    //READ H and normalize
    int HaddX = LSIZE_X; int HaddY = LSIZE_Y;
    if ((x + HaddX) < LSIZE_MEM_X && (xG + HaddX) < VOL_SIZE_X
    && (y + HaddY) < LSIZE_MEM_Y && (yG + HaddY) < VOL_SIZE_Y){
    int locGlobalH = ((xG + HaddX) + (yG + HaddY)*VOL_SIZE_X + (zG + addZ)*VOL_SIZE_XtY); //Component later
    float valH = readAndNormalize(volume, locGlobalH, -1.0f);
    setLocalValue(localStorage, (x + HaddX), (y + HaddY), (z + addZ), valH);
    }
    }
    }*/
    if (voxelValue < -0.5f){//true){
        //MEMBLOCK //REALLY IMPORTANT ONE
        //barrier(CLK_LOCAL_MEM_FENCE);
        //All data is read to local, perform calculation

        float accumulationValue = 0.0f;
        int counter = 0;
        int HW_boost = 0;
        while ((counter == 0) && (PROGRESSIVE_PNN || (HW_boost == 0)) && (HW_boost < 3)){
            //int minX = xG - HALF_WIDTH - HW_boost; //or max this and 0? or sampler handles it?
            int minX = max2i((xG - HALF_WIDTH - HW_boost), 0); //x; //or max this and 0? or sampler handles it?
            //int minY = yG - HALF_WIDTH - HW_boost;
            int minY = max2i((yG - HALF_WIDTH - HW_boost), 0); //yG - HALF_WIDTH - HW_boost; //y;
            //int minZ = zG - HALF_WIDTH - HW_boost;
            int minZ = max2i((zG - HALF_WIDTH - HW_boost), 0); //zG - HALF_WIDTH - HW_boost; //z;
            int maxX = minX + HALF_WIDTH_X2 + (2 * HW_boost);
            maxX = min2i(maxX, ((int)VOL_SIZE_X - 1));
            int maxY = minY + HALF_WIDTH_X2 + (2 * HW_boost);
            maxY = min2i(maxY, ((int)VOL_SIZE_Y - 1)); // minY + HALF_WIDTH_X2 + (2 * HW_boost);
            int maxZ = minZ + HALF_WIDTH_X2 + (2 * HW_boost);
            maxZ = min2i(maxZ, ((int)VOL_SIZE_Z - 1)); //minZ + HALF_WIDTH_X2 + (2 * HW_boost);

            //Can restructure to avoid overlap!
            //  Starts at 0 HW and adds up to HALF_WIDTH(max) ie. 0,1,2,3 for GridSize 7
            // 0 means checking the node itself
            for (int xi = minX; xi <= maxX; xi++){
                for (int yi = minY; yi <= maxY; yi++){
                    for (int zi = minZ; zi <= maxZ; zi++){
                        if (xi == xG && yi == yG && zi == zG){ continue; }
                        //int loc = (xi + yi*LSIZE_MEM_X + zi*LSIZE_MEM_XtY);
                        int loc = (xi + yi*VOL_SIZE_X + zi*VOL_SIZE_XtY);
                        //float locValue = localStorage[loc];
                        float locValue = readAndNormalize(volume, loc, -1.0f);
                        if (locValue >= -0.5f){ //ev > -0.5? for inaccuracy?
                            accumulationValue += locValue;
                            counter++;
                        }
                    }
                }
            }
            HW_boost++;
        }

        //float 
        if (counter != 0 && accumulationValue >= 0.0f){
            voxelValue = accumulationValue / counter;
        }
        else {
            voxelValue = 0.0f;
        }

    }

    //barrier(CLK_LOCAL_MEM_FENCE);
    int outputDataType = get_image_channel_data_type(outputVolume);
    if (outputDataType == CLK_FLOAT) {
        write_imagef(outputVolume, pos, voxelValue);
    }
    else if (outputDataType == CLK_UNSIGNED_INT8 || outputDataType == CLK_UNSIGNED_INT16) {
        write_imageui(outputVolume, pos, round(voxelValue));
    }
    else {
        write_imagei(outputVolume, pos, round(voxelValue));
    }
}