## Dataset Overview
- **Dataset Name**: Photorealistic Faces with Different Facial Action Units
- **Data Access**: [Google Drive](https://drive.google.com/drive/folders/1wzqoBauX746f9YxpFrmf8TUlUhfb8vDN?usp=sharing)

![Screenshot_20221220_095402](https://user-images.githubusercontent.com/24949723/208733705-827a0670-f3d7-4d31-9a87-f66cb483a4e4.png) 

## A Sample Image 
- **An European Male of Skin Color Scale 4 with AU10 Activated at its Maximum**:
![Screenshot_20221220_095745](https://user-images.githubusercontent.com/24949723/208734519-b632191e-9ba3-4a45-86eb-3e4674c35cd7.png)

## Softwares and Websites Used
- Character Creator 4 (CC4)
- Power Automate
- https://skintone.google/ 

## Skin Color Scale from Google
![Screenshot_20221223_062225](https://user-images.githubusercontent.com/24949723/209418054-f2680cc0-6e95-4f88-a862-0ce58d3f394c.png)

## Composition
- **Calculation**: (9 AUs * 5 Intensity Levels + 1 AU43 * 1 Intensity Level + Original Face without AU) * 10 Skin Colors * (1 European Male Face + 1 European Female Face) = 940 Samples

## Details
Each sample features only one action unit activated while other AUs remain unactivated. The action units include:
- **AU[-]-[AU's name]**: [Corresponding facial morphing options in CC4 with activation power levels]
- **AU4-Brow Lowerer**: `Brow Drop L/R (30, 60, 90, 120, 150)`
- **AU6-Cheek Raiser**: `Cheek Raise L/R (30, 60, 90, 120, 150)`
- **AU7-Lid Tightener**: `Eye Squint L/R (30, 60, 90, 120, 150)`
- **AU9-Nose Wrinkler**: `Nose Sneer L/R (30, 60, 90, 120, 150)`
- **AU10-Upper Lip Raiser**: `Nose Nostril Raise L/R (30, 60, 90, 120, 150)`, `Nose Crease L/R (20, 40, 60, 80, 100)`, `Mouth Shrug Upper (30, 60, 90, 120, 150)`
- **AU12-Lip Corner Puller**: `Mouth Smile L/R (30, 60, 90, 120, 150)`
- **AU20-Lip Stretcher**: `Mouth Stretch L/R (30, 60, 90, 120, 150)`
- **AU25-Lips Part**: `Mouth Shrug Upper (16, 32, 48, 64, 80)`, `Mouth Drop Lower (16, 32, 48, 64, 80)`
- **AU26-Jaw Drop**: `Jaw Open (10, 20, 30, 40, 50)`
- **AU43-Eyes Closed**: `Eye Blink L/R (100)`
