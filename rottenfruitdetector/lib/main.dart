import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(ImageInputApp());
}

class ImageInputApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Input App',
      home: ImageInputScreen(),
    );
  }
}

class ImageInputScreen extends StatefulWidget {
  @override
  _ImageInputScreenState createState() => _ImageInputScreenState();
}

class _ImageInputScreenState extends State<ImageInputScreen> {
  File? _imageFile; // To store the selected image file

  final picker = ImagePicker(); // Instance of ImagePicker for image selection

  Future getImage() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _imageFile = File(pickedFile.path); // Store the picked image file
      } else {
        print('No image selected.');
      }
    });
  }

  void navigateToResultPage() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => ResultPage(imageFile: _imageFile)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Input'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Center(
            child: _imageFile == null
                ? Text('No image selected.')
                : _getImageWidget(_imageFile!), // Display selected image or message
          ),
          ElevatedButton(
            onPressed: navigateToResultPage,
            child: Text('Go to Result'),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: getImage,
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }

  // Method to display the image based on platform (web or non-web)
  Widget _getImageWidget(File imageFile) {
    if (kIsWeb) {
      // Display image using Image.network() for web platform
      return Image.network(imageFile.path);
    } else {
      // Display image using Image.file() for non-web platforms
      return Image.file(imageFile);
    }
  }
}

class ResultPage extends StatelessWidget {
  final File? imageFile;

  const ResultPage({Key? key, this.imageFile}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Result'),
      ),
      body: Center(
        child: imageFile == null
            ? Text('No image selected.')
            : _getImageWidget(imageFile!), // Display the image based on platform
      ),
    );
  }

  // Method to display the image based on platform (web or non-web)
  Widget _getImageWidget(File imageFile) {
    if (kIsWeb) {
      // Display image using Image.network() for web platform
      return Image.network(imageFile.path);
    } else {
      // Display image using Image.file() for non-web platforms
      return Image.file(imageFile);
    }
  }
}
