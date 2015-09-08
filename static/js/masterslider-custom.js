/* 
 * Bizwrap v1.4
 * Design_mylife
 */


var slider = new MasterSlider();

slider.control('arrows', {insertTo: '#masterslider'});
slider.control('bullets');

slider.setup('masterslider', {
    width: 1024,
    height: 600,
    space: 5,
    view: 'basic',
    layout: 'fullwidth',
    preload: 0,
    instantStartLayers: true,
    autoplay: true
});

//team carousel


		var slider = new MasterSlider();
		slider.setup('teamslider' , {
			loop:true,
			width:240,
			height:240,
			speed:20,
			view:'fadeBasic',
			preload:0,
			space:0,
			wheel:true
		});
		slider.control('arrows');
		slider.control('slideinfo',{insertTo:'#staff-info'});