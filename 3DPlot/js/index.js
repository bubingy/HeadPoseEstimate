let scene,camera,renderer,axes,controls

init();
animate();

function createGeometry()
{
    let pointSet = []
    for(point of data["points"]) {
        pointSet.push(new THREE.Vector3(point[0],point[1],point[2])); 
    }
    let setgeometry = new THREE.BufferGeometry().setFromPoints(pointSet);
    let setmaterial = new THREE.PointsMaterial(
        { color : 0xff0000 , size : 10 ,sizeAttenuation : false}
    );
    let plot = new THREE.Points( setgeometry , setmaterial );

    scene.add(plot);

    for (arrow of data["arrows"]) {
        let arrowPosition = new THREE.Vector3(
            arrow["position"][0],
            arrow["position"][1],
            arrow["position"][2]
        );
        let arrowDirection = new THREE.Vector3(
            arrow["direction"][0],
            arrow["direction"][1],
            arrow["direction"][2]
        );
        arrowHelper = new THREE.ArrowHelper( 
            arrowDirection, arrowPosition, 20, 0xffff00, 0.25, 0.08 
        );
        scene.add(arrowHelper);
    }
}    

function init(){
    scene = new THREE.Scene;
    scene.background = new THREE.Color( 0xcccccc);

    camera = new THREE.PerspectiveCamera(25 , window.innerWidth/window.innerHeight , 1 , 1000);
    camera.position.set( 500, 500, 500 );
    let axes =new THREE.AxesHelper(200);
    scene.add(axes);

    createGeometry();

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth,window.innerHeight);
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls( camera, renderer.domElement );

    controls.screenSpacePanning = false;

    controls.minDistance = 10;
    controls.maxDistance = 1000;

    window.addEventListener( 'resize', onWindowResize, false );
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
}

function animate() {
    requestAnimationFrame( animate );
    controls.update();
    render();
}

function render() {
    renderer.render( scene, camera );
}
