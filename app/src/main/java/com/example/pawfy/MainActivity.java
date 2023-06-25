package com.example.pawfy;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.pawfy.ml.Model;
import com.google.flatbuffers.ByteBufferUtil;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;

public class MainActivity extends AppCompatActivity {
    Button camera, gallery;
    ImageView imageView;
    TextView result, details;
    int imageSize = 224;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        details = findViewById(R.id.details);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }
                else{
                    requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer  byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for(int i=0 ; i < imageSize ; i ++){
                for(int j=0 ; j < imageSize ; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }


            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            //find the index of the class with the biggest confidence
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i=0 ; i < confidences.length; i++){
                if(confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {/*Abyssinian*/"Lifespan: 9-15 years\n" +
                    "Weight: 6-10 lbs\n" +
                    "Traits: sleek and muscular build, almond-shaped eyes, large ears, ticked coat with distinct bands of color\n" +
                    "Behavior: active, intelligent, curious, affectionate\n" +
                    "Common diseases: periodontal disease, kidney disease, hyperthyroidism\n",
                    /*Bengal Cat*/"Lifespan: 10-16 years\n" +
                    "Weight: 8-15 lbs\n" +
                    "Traits: muscular build, distinctive spotted or marbled coat, large, expressive eyes\n" +
                    "Behavior: active, playful, intelligent, curious, social\n" +
                    "Common diseases: progressive retinal atrophy, hypertrophic cardiomyopathy, cataracts",
                    /*Birman*/"Lifespan: 12-16 years\n" +
                    "Weight: 6-12 lbs\n" +
                    "Traits: blue eyes, white feet, silky coat with colorpoint pattern\n" +
                    "Behavior: gentle, affectionate, social, adaptable\n" +
                    "Common diseases: hypertrophic cardiomyopathy, kidney disease, gingivitis",
                    /*Bombay*/"Lifespan: 12-16 years\n" +
                    "Weight: 6-11 lbs\n" +
                    "Traits: black coat, muscular build, large, round eyes\n" +
                    "Behavior: intelligent, affectionate, social, vocal\n" +
                    "Common diseases: hypertrophic cardiomyopathy, periodontal disease, obesity\n",
                    /*British Shorthair*/"Lifespan: 12-17 years\n" +
                    "Weight: 9-18 lbs\n" +
                    "Traits: compact, muscular build, round face and eyes, dense, plush coat in various colors\n" +
                    "Behavior: calm, affectionate, independent, adaptable\n" +
                    "Common diseases: polycystic kidney disease, hypertrophic cardiomyopathy, obesity\n",
                    /*Egyptian Mau*/"Lifespan: 12-16 years\n" +
                    "Weight: 6-14 lbs\n" +
                    "Traits: spotted coat with \"M\" marking on forehead, large, green eyes, muscular build\n" +
                    "Behavior: active, playful, intelligent, loyal\n" +
                    "Common diseases: retinal dysplasia, patellar luxation, pyruvate kinase deficiency\n",
                    /*Maine Coon*/"Lifespan: 12-15 years\n" +
                    "Weight: 9-18 lbs\n" +
                    "Traits: large, muscular build, fluffy coat in various colors, tufted ears, bushy tail\n" +
                    "Behavior: gentle, friendly, sociable, intelligent\n" +
                    "Common diseases: hypertrophic cardiomyopathy, hip dysplasia, spinal muscular atrophy\n",
                    /*Persian*/"Lifespan: 12-16 years\n" +
                    "Weight: 7-12 lbs\n" +
                    "Traits: long, fluffy coat, flat face, round eyes, short legs\n" +
                    "Behavior: calm, affectionate, independent, low-maintenance\n" +
                    "Common diseases: polycystic kidney disease, hypertrophic cardiomyopathy, dental issues\n",
                    /*Ragdoll*/"Lifespan: 12-17 years\n" +
                    "Weight: 10-20 lbs\n" +
                    "Traits: large, muscular build, long, plush coat in various colors, blue eyes\n" +
                    "Behavior: docile, friendly, affectionate, loyal\n" +
                    "Common diseases: hypertrophic cardiomyopathy, bladder stones, gingivitis\n",
                    /*Russian Blue*/"Lifespan: 10-16 years\n" +
                    "Weight: 7-12 lbs\n" +
                    "Traits: short, dense coat with silver-blue color, green eyes, elegant build\n" +
                    "Behavior: intelligent, independent, shy, reserved\n" +
                    "Common diseases: hypertrophic cardiomyopathy, patellar luxation, bladder stones\n",
                    /*Siamese*/"Lifespan: 12-16 years\n" +
                    "Weight: 8-12 lbs\n" +
                    "Traits: sleek, muscular build, pointed coat with blue eyes, large ears\n" +
                    "Behavior: active, vocal, intelligent, social\n" +
                    "Common diseases: dental issues, respiratory infections, amyloidosis\n",
                    /*Sphynx*/"Lifespan: 8-14 years\n" +
                    "Weight: 6-12 lbs\n" +
                    "Traits: hairless, wrinkled skin, large ears, almond-shaped eyes\n" +
                    "Behavior: energetic, curious, affectionate, social\n" +
                    "Common diseases: hypertrophic cardiomyopathy, respiratory infections, periodontal disease.",
                    /*american bulldog*/"Lifespan: 10-16 years\n" +
                    "Weight: 60-120 lbs\n" +
                    "Traits: muscular build, short coat in various colors, powerful jaws\n" +
                    "Behavior: loyal, protective, energetic, confident\n" +
                    "Common diseases: hip dysplasia, skin allergies, bloat\n",
                    /*american pit bull terrier*/"Lifespan: 8-15 years\n" +
                    "Weight: 30-65 lbs\n" +
                    "Traits: muscular build, short coat in various colors, strong jaws\n" +
                    "Behavior: loyal, affectionate, energetic, strong-willed\n" +
                    "Common diseases: hip dysplasia, skin allergies, heart disease\n",
                    /*basset hound*/"Lifespan: 10-12 years\n" +
                    "Weight: 40-65 lbs\n" +
                    "Traits: long, droopy ears, short legs, loose skin, short coat in various colors\n" +
                    "Behavior: laid-back, friendly, loyal, stubborn\n" +
                    "Common diseases: hip dysplasia, ear infections, bloat\n",
                    /*beagle*/"Lifespan: 12-15 years\n" +
                    "Weight: 18-30 lbs\n" +
                    "Traits: long, droopy ears, short coat in various colors, compact build\n" +
                    "Behavior: friendly, curious, energetic, stubborn\n" +
                    "Common diseases: hip dysplasia, ear infections, epilepsy\n",
                    /*boxer*/"Lifespan: 10-12 years\n" +
                    "Weight: 50-80 lbs\n" +
                    "Traits: muscular build, short coat in various colors, square-shaped head\n" +
                    "Behavior: loyal, playful, energetic, protective\n" +
                    "Common diseases: hip dysplasia, heart disease, cancer\n",
                    /*chihuahua*/"Lifespan: 12-20 years\n" +
                    "Weight: 2-6 lbs\n" +
                    "Traits: small size, large ears, short or long coat in various colors\n" +
                    "Behavior: alert, feisty, loyal, affectionate\n" +
                    "Common diseases: dental problems, heart disease, patellar luxation\n",
                    /*english cocker spaniel*/"english cocker sLifespan: 12-15 years\n" +
                    "Weight: 26-34 lbs\n" +
                    "Traits: long, droopy ears, silky coat in various colors, friendly expression\n" +
                    "Behavior: friendly, energetic, loyal, gentle\n" +
                    "Common diseases: hip dysplasia, ear infections, eye problems\n",
                    /*english setter*/"Lifespan: 10-12 years\n" +
                    "Weight: 45-80 lbs\n" +
                    "Traits: long, silky coat in various colors, friendly expression, athletic build\n" +
                    "Behavior: friendly, affectionate, intelligent, energetic\n" +
                    "Common diseases: hip dysplasia, bloat, deafness\n",
                    /*german shorthaired*/"Lifespan: 12-14 years\n" +
                    "Weight: 45-70 lbs\n" +
                    "Traits: short, dense coat in various colors, athletic build\n" +
                    "Behavior: friendly, intelligent, energetic, loyal\n" +
                    "Common diseases: hip dysplasia, eye problems, bloat\n",
                    /*great pyrenees*/"Lifespan: 10-12 years\n" +
                    "Weight: 85-115 lbs\n" +
                    "Traits: thick, white coat, large size, powerful build\n" +
                    "Behavior: loyal, protective, gentle, independent\n" +
                    "Common diseases: hip dysplasia, bloat, eye problems",
                    /*havanese*/"Lifespan: 14-16 years\n" +
                    "Weight: 7-13 lbs\n" +
                    "Traits: long, silky coat in various colors, small size\n" +
                    "Behavior: friendly, affectionate, playful, intelligent\n" +
                    "Common diseases: patellar luxation, dental problems, eye problems\n",
                    /*japanese chin*/"Lifespan: 10-12 years\n" +
                    "Weight: 4-9 lbs\n" +
                    "Traits: short, silky coat in various colors, large, expressive eyes\n" +
                    "Behavior: affectionate, loyal, calm, adaptable\n" +
                    "Common diseases: heart disease, eye problems, dental problems\n",
                    /*keeshond*/"Lifespan: 12-15 years\n" +
                    "Weight: 35-45 lbs\n" +
                    "Traits: thick, double coat in various colors, fox-like face\n" +
                    "Behavior: friendly, affectionate, playful, intelligent\n" +
                    "Common diseases: hip dysplasia, skin allergies, heart disease\n",
                    /*leonberger*/"Lifespan: 7-9 years\n" +
                    "Weight: 120-170 lbs\n" +
                    "Traits: thick, double coat in various colors, large size\n" +
                    "Behavior: friendly, calm, loyal, protective\n" +
                    "Common diseases: hip dysplasia, heart disease, bloat\n",
                    /*miniature pinscher*/"Lifespan: 12-16 years\n" +
                    "Weight: 8-10 lbs\n" +
                    "Traits: short, smooth coat in various colors, small size, high energy\n" +
                    "Behavior: lively, alert, loyal, independent\n" +
                    "Common diseases: patellar luxation, eye problems, dental problems\n",
                    /*newfoundland*/"Lifespan: 8-10 years\n" +
                    "Weight: 100-150 lbs\n" +
                    "Traits: thick, water-resistant coat in various colors, large size\n" +
                    "Behavior: friendly, gentle, loyal, calm\n" +
                    "Common diseases: hip dysplasia, heart disease, bloat\n",
                    /*pomeranian*/"Lifespan: 12-16 years\n" +
                    "Weight: 3-7 lbs\n" +
                    "Traits: fluffy coat in various colors, small size, fox-like face\n" +
                    "Behavior: lively, friendly, affectionate, independent\n" +
                    "Common diseases: dental problems, patellar luxation, tracheal collapse",
                    /*pug*/"Lifespan: 12-15 years\n" +
                    "Weight: 14-18 lbs\n" +
                    "Traits: wrinkled face, short coat in various colors, compact build\n" +
                    "Behavior: friendly, affectionate, lazy, stubborn\n" +
                    "Common diseases: breathing problems, eye problems, skin allergies\n",
                    /*saint bernard*/"Lifespan: 8-10 years\n" +
                    "Weight: 120-180 lbs\n" +
                    "Traits: thick, fluffy coat in various colors, large size\n" +
                    "Behavior: friendly, gentle, loyal, calm\n" +
                    "Common diseases: hip dysplasia, bloat, skin allergies",
                    /*samoyed*/"Age: 12-14 years\n" +
                    "Weight: 35-65 pounds\n" +
                    "Traits: Friendly, playful, and active\n" +
                    "Behavior: Samoyeds are known for their affectionate nature and love to be around their family. They are also active and energetic, requiring daily exercise and mental stimulation.\n" +
                    "Common diseases: Hip dysplasia, diabetes, eye diseases, allergies\n",
                    /*scottish terrier*/"Age: 11-13 years\n" +
                    "Weight: 18-22 pounds\n" +
                    "Traits: Independent, feisty, and loyal\n" +
                    "Behavior: Scottish Terriers are independent and often stubborn, but also fiercely loyal to their family. They have a strong prey drive and may not get along well with other pets.\n" +
                    "Common diseases: Von Willebrand's disease, skin allergies, bladder cancer\n",
                    /*shiba inu*/"Age: 12-15 years\n" +
                    "Weight: 17-23 pounds\n" +
                    "Traits: Alert, confident, and independent\n" +
                    "Behavior: Shiba Inus are independent and confident dogs that may not be very affectionate. They are also very alert and make excellent watchdogs.\n" +
                    "Common diseases: Hip dysplasia, allergies, eye diseases\n",
                    /*staffordshire bull terrier*/"Age: 12-14 years\n" +
                    "Weight: 24-38 pounds\n" +
                    "Traits: Affectionate, courageous, and loyal\n" +
                    "Behavior: Staffordshire Bull Terriers are affectionate and loyal to their family, but may not get along well with other pets. They are also courageous and make excellent guard dogs.\n" +
                    "Common diseases: Hip dysplasia, skin allergies, heart disease\n",
                    /*wheaten terrier*/"Age: 12-15 years\n" +
                    "Weight: 35-45 pounds\n" +
                    "Traits: Friendly, energetic, and loyal\n" +
                    "Behavior: Wheaten Terriers are friendly and energetic dogs that love to be around their family. They are also intelligent and trainable.\n" +
                    "Common diseases: Protein-losing nephropathy, hip dysplasia, skin allergies\n",
                    /*yorkshire terrier*/"Age: 12-15 years\n" +
                    "Weight: 4-7 pounds\n" +
                    "Traits: Confident, bold, and affectionate\n" +
                    "Behavior: Yorkshire Terriers are confident and bold dogs that may not be suitable for families with small children. They are also very affectionate and make great lap dogs.\n" +
                    "Common diseases: Portosystemic shunt, tracheal collapse, dental problems"};

            String[] breeds = {"Abyssinian", "Bengal Cat", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
                    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
                    "american bulldog", "american pit bull terrier", "basset hound", "beagle", "boxer", "chihuahua",
                    "english cocker spaniel", "english_setter", "german_shorthaired", "great pyrenees", "havanese", "japanese chin",
                    "keeshond", "leonberger", "miniature pinscher", "newfoundland", "pomeranian", "pug",
                    "saint bernard", "samoyed", "scottish terrier", "shiba inu", "staffordshire bull terrier", "wheaten terrier",
                    "yorkshire terrier"};
            result.setText(breeds[maxPos]);
            details.setText(classes[maxPos]);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try{
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}