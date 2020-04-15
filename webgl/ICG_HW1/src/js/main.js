import { mat4 } from "gl-matrix"
import WebGLUtils from "./webgl-utils.js"

import startVertexShader from "../shaders/phongVertexShader.glsl"
import startFragmentShader from "../shaders/phongFragmentShader.glsl"
import teapotModel from "../model/Teapot.json"

// common variables
var gl;
var shaderProgram;

var mvMatrix = mat4.create();
var pMatrix  = mat4.create();

var teapotVertexPositionBuffer;
var teapotVertexNormalBuffer;
var teapotVertexFrontColorBuffer;

var teapotAngle = 180;
var lastTime    = 0;

function initGL(canvas) {
    try {
        gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        gl.viewportWidth  = canvas.width;
        gl.viewportHeight = canvas.height;
    } 
    catch (e) {
    }

    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(");
    }
}

function getShader(shaderSource, type) {
    var shader;
    if (type == "fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } 
    else if (type == "vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } 
    else {
        return null;
    }

    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function initShaders() {
    var fragmentShader = getShader(startFragmentShader, "fragment");
    var vertexShader   = getShader(startVertexShader, "vertex");

    shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(shaderProgram);

    shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

  shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
  gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

    shaderProgram.vertexFrontColorAttribute = gl.getAttribLocation(shaderProgram, "aFrontColor");
    gl.enableVertexAttribArray(shaderProgram.vertexFrontColorAttribute);

    shaderProgram.pMatrixUniform  = gl.getUniformLocation(shaderProgram, "uPMatrix");
    shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");

  shaderProgram.pointLightPos1 = gl.getUniformLocation(shaderProgram, "u_pointLightPos1");
  shaderProgram.lightColor1 = gl.getUniformLocation(shaderProgram, "u_lightColor1");
  shaderProgram.pointLightPos2 = gl.getUniformLocation(shaderProgram, "u_pointLightPos2");
  shaderProgram.lightColor2 = gl.getUniformLocation(shaderProgram, "u_lightColor2");
  shaderProgram.ambientColor = gl.getUniformLocation(shaderProgram, "u_ambient");
  shaderProgram.diffuseColor = gl.getUniformLocation(shaderProgram, "u_diffuse");
  shaderProgram.specularColor = gl.getUniformLocation(shaderProgram, "u_specular");
  shaderProgram.shiningness = gl.getUniformLocation(shaderProgram, "u_shiningness");
}

function setMatrixUniforms() {
    gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, pMatrix);
    gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, mvMatrix);
}

function degToRad(degrees) {
    return degrees * Math.PI / 180;
}

function handleLoadedTeapot(teapotData) {
    teapotVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(teapotData.vertexPositions), gl.STATIC_DRAW);
    teapotVertexPositionBuffer.itemSize = 3;
    teapotVertexPositionBuffer.numItems = teapotData.vertexPositions.length / 3;

    teapotVertexNormalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexNormalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(teapotData.vertexNormals), gl.STATIC_DRAW);
    teapotVertexNormalBuffer.itemSize = 3;
    teapotVertexNormalBuffer.numItems = teapotData.vertexNormals.length / 3;

    teapotVertexFrontColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexFrontColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(teapotData.vertexFrontcolors), gl.STATIC_DRAW);
    teapotVertexFrontColorBuffer.itemSize = 3;
    teapotVertexFrontColorBuffer.numItems = teapotData.vertexFrontcolors.length / 3;
}

function loadTeapot() {
  handleLoadedTeapot(teapotModel);
}

/*
    TODO HERE:
    add two or more objects showing on the canvas
    (it needs at least three objects showing at the same time)
*/
function drawScene() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (teapotVertexPositionBuffer   == null || 
        teapotVertexNormalBuffer     == null || 
        teapotVertexFrontColorBuffer == null) {
        
        return;
    }

    // Setup Projection Matrix
    mat4.perspective(pMatrix, 45, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0);

    // Setup Model-View Matrix
    mat4.identity(mvMatrix);
    mat4.translate(mvMatrix, mvMatrix, [0, 0, -40]);
    mat4.rotate(mvMatrix, mvMatrix, degToRad(teapotAngle), [0, 1, 0]);

    setMatrixUniforms();

  gl.uniform3f(shaderProgram.pointLightPos1, -8.0, -3.0, 1.0);
  gl.uniform3f(shaderProgram.lightColor1, 0.5, 0.5, 0.5);
  gl.uniform3f(shaderProgram.pointLightPos2, 8.0, 3.0, 1.0);
  gl.uniform3f(shaderProgram.lightColor2, 0.5, 0.5, 0.5);
  gl.uniform3f(shaderProgram.ambientColor, 0.24, 0.19, 0.07);
  gl.uniform3f(shaderProgram.diffuseColor, 0.75, 0.6, 0.22);
  gl.uniform3f(shaderProgram.specularColor, 0.62, 0.55, 0.36);
  gl.uniform1f(shaderProgram.shiningness, 0.4 * 128.0);

    // Setup teapot position data
    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexPositionBuffer);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 
                           teapotVertexPositionBuffer.itemSize, 
                           gl.FLOAT, 
                           false, 
                           0, 
                           0);

    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexNormalBuffer);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 
                           teapotVertexNormalBuffer.itemSize, 
                           gl.FLOAT, 
                           false, 
                           0, 
                           0);

    // Setup teapot front color data
    gl.bindBuffer(gl.ARRAY_BUFFER, teapotVertexFrontColorBuffer);
    gl.vertexAttribPointer(shaderProgram.vertexFrontColorAttribute, 
                           teapotVertexFrontColorBuffer.itemSize, 
                           gl.FLOAT, 
                           false, 
                           0, 
                           0);

    gl.drawArrays(gl.TRIANGLES, 0, teapotVertexPositionBuffer.numItems);
}

function animate() {
    var timeNow = new Date().getTime();
    if (lastTime != 0) {
        var elapsed = timeNow - lastTime;
        teapotAngle += 0.03 * elapsed;
    }
    
    lastTime = timeNow;
}

function tick() {
    requestAnimFrame(tick);
    drawScene();
    animate();
}

function webGLStart() {
    var canvas = document.getElementById("ICG-canvas");
    initGL(canvas);
    initShaders();
    loadTeapot();

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);

    tick();
}

window.onload = function() {
  webGLStart();
}