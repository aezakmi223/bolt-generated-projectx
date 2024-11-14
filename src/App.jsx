import { useEffect, useRef, useState } from 'react'
import * as faceapi from 'face-api.js'
import { Button, Container, Typography, Box, CircularProgress, Paper } from '@mui/material'
import { PhotoCamera } from '@mui/icons-material'

function App() {
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [imageURL, setImageURL] = useState(null)
  const [symmetryScore, setSymmetryScore] = useState(null)
  const imageRef = useRef()
  const canvasRef = useRef()

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    const MODEL_URL = '/models'
    
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
      ])
      setIsModelLoading(false)
    } catch (error) {
      console.error('Error loading models:', error)
    }
  }

  const calculateSymmetry = async (face) => {
    const landmarks = face.landmarks
    const positions = landmarks.positions

    // Calculate symmetry based on key facial landmarks
    let totalDifference = 0
    const pairs = [
      [0, 16], // Jaw points
      [1, 15],
      [2, 14],
      [3, 13],
      [4, 12],
      [5, 11],
      [6, 10],
      [7, 9],
      [37, 43], // Eyes
      [38, 42],
      [39, 41],
      [40, 40],
      [31, 35], // Nose
    ]

    pairs.forEach(([left, right]) => {
      const leftPoint = positions[left]
      const rightPoint = positions[right]
      const diff = Math.abs(leftPoint.x - rightPoint.x) + Math.abs(leftPoint.y - rightPoint.y)
      totalDifference += diff
    })

    // Convert to a 0-100 score (lower difference means higher symmetry)
    const maxDifference = 100 // Arbitrary maximum difference
    const score = Math.max(0, 100 - (totalDifference / maxDifference * 100))
    return Math.round(score)
  }

  const handleImageUpload = async (e) => {
    const file = e.target.files[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setImageURL(url)
      setSymmetryScore(null)

      const img = await faceapi.bufferToImage(file)
      imageRef.current.src = img.src
      
      imageRef.current.onload = async () => {
        const detections = await faceapi
          .detectAllFaces(imageRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()

        if (detections.length > 0) {
          const score = await calculateSymmetry(detections[0])
          setSymmetryScore(score)

          // Draw face landmarks
          const canvas = canvasRef.current
          canvas.width = imageRef.current.width
          canvas.height = imageRef.current.height
          const ctx = canvas.getContext('2d')
          faceapi.draw.drawFaceLandmarks(canvas, detections)
        }
      }
    }
  }

  return (
    <Container maxWidth="md">
      <Typography variant="h3" gutterBottom>
        Face Symmetry Analyzer
      </Typography>

      {isModelLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', m: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Button
            variant="contained"
            component="label"
            startIcon={<PhotoCamera />}
            sx={{ mb: 4 }}
          >
            Upload Photo
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={handleImageUpload}
            />
          </Button>

          {imageURL && (
            <Paper elevation={3} sx={{ p: 2, mb: 4 }}>
              <div className="canvas-wrapper" style={{ width: 'fit-content' }}>
                <img
                  ref={imageRef}
                  src={imageURL}
                  alt="Uploaded face"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
                <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0 }} />
              </div>

              {symmetryScore !== null && (
                <Typography variant="h4" sx={{ mt: 2 }}>
                  Symmetry Score: {symmetryScore}%
                </Typography>
              )}
            </Paper>
          )}
        </>
      )}
    </Container>
  )
}

export default App
