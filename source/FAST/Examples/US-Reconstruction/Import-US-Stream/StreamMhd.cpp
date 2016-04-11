/**
 * Examples/DataImport/streamImagesFromDisk.cpp
 *
 * If you edit this example, please also update the wiki and source code file in the repository.
 */
#include "FAST/Streamers/ImageFileStreamer.hpp"
#include "FAST/Visualization/ImageRenderer/ImageRenderer.hpp"
#include "FAST/Visualization/SimpleWindow.hpp"
#include "FAST/TestDataPath.hpp"

using namespace fast;

int main() {
    // Import images from files using the ImageFileStreamer
    ImageFileStreamer::pointer streamer = ImageFileStreamer::New();
    // The hashtag here will be replaced with an integer, starting with 0 as default
    //std::string folder = 'US-2Dt';
    //std::nameformat = 'US-2Dt_#.mhd';

    std::string folder = "/rekonstruksjons_data/US_01_20130529T084519/";
    std::string nameformat = "US_01_20130529T084519_ScanConverted_#.mhd";
    
    streamer->setFilenameFormat(std::string(FAST_TEST_DATA_DIR)+folder+nameformat);

    // Renderer image
    ImageRenderer::pointer renderer = ImageRenderer::New();
    renderer->addInputConnection(streamer->getOutputPort());
    SimpleWindow::pointer window = SimpleWindow::New();
    window->addRenderer(renderer);
    window->set2DMode();
    window->setTimeout(5*1000); // automatically close window after 5 seconds
    window->start();
}