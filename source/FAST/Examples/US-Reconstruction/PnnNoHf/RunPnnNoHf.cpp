/**
 * Examples/DataImport/streamImagesFromDisk.cpp
 *
 * If you edit this example, please also update the wiki and source code file in the repository.
 */
#include "FAST/Streamers/ImageFileStreamer.hpp"
#include "FAST/Visualization/ImageRenderer/ImageRenderer.hpp"
#include "FAST/Visualization/VolumeRenderer/VolumeRenderer.hpp"
#include "FAST/Visualization/SimpleWindow.hpp"
#include "FAST/TestDataPath.hpp"
#include "FAST/Algorithms/UsReconstruction/PnnNoHf.hpp"

using namespace fast;

int main() {
    // Import images from files using the ImageFileStreamer
    ImageFileStreamer::pointer streamer = ImageFileStreamer::New();
    // The hashtag here will be replaced with an integer, starting with 0 as default
    //std::string folder = 'US-2Dt';
    //std::nameformat = 'US-2Dt_#.mhd';

    std::string folder = "/rekonstruksjons_data/US_01_20130529T084519/";
    std::string nameformat = "US_01_20130529T084519_ScanConverted_#.mhd";
    
    streamer->setStreamingMode(STREAMING_MODE_PROCESS_ALL_FRAMES);
    //streamer->setStreamingMode(STREAMING_MODE_STORE_ALL_FRAMES);
    streamer->setFilenameFormat(std::string(FAST_TEST_DATA_DIR)+folder+nameformat);
    streamer->setMaximumNumberOfFrames(200);
    std::cout << "Nr of frames" << streamer->getNrOfFrames() << std::endl;

    // Reconstruction PNN
    PnnNoHf::pointer pnn = PnnNoHf::New();
    pnn->setInputConnection(streamer->getOutputPort());

    //Alt for now, display image
    ImageRenderer::pointer imageRenderer = ImageRenderer::New();
    imageRenderer->addInputConnection(pnn->getOutputPort());

    // Renderer volume
    //VolumeRenderer::pointer renderer = VolumeRenderer::New();
    //ImageRenderer::pointer renderer = ImageRenderer::New();
    //renderer->addInputConnection(pnn->getOutputPort());
    SimpleWindow::pointer window = SimpleWindow::New();
    window->addRenderer(imageRenderer);//renderer);
    //window->setMaximumFramerate(10); //unngå at buffer går tomt?
    //window->set2DMode();
    window->setTimeout(5*1000); // automatically close window after 5 seconds
    window->start();
}