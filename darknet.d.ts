/// <reference types="node" />
export declare class Darknet {
    darknet: any;
    meta: any;
    netPointer: any;
    net: any;
    netSize: number;
    names: string[];
    private memoryIndex;
    private memorySlotsUsed;
    private memoryCount;
    private memory;
    /**
     * A new instance of pjreddie's darknet. Create an instance as soon as possible in your app, because it takes a while to init.
     * @param config
     */
    constructor(config: IDarknetConfig);
    dispose(): void;
    letterboxImage(image: Image): Promise<Image>;
    freeImage(image: Image): void;
    freeDetection(dets: any, num: number): void;
    resetMemory({ memory }?: {
        memory?: number;
    }): void;
    private freeMemory;
    private makeMemory;
    private rememberNet;
    private avgPrediction;
    private getArrayFromBuffer;
    private bufferToDetections;
    detection(letterBoxImage: any, w: number, h: number, thresh: number, hier: number): Promise<{
        dets: Buffer;
        num: number;
    }>;
    NMS(dets: Buffer, num: number, nms: number): Promise<void>;
    interpretation(dets: Buffer, num: number): Detection[];
    protected _detectAsync(net: any, meta: any, image: any, thresh?: number, hier_thresh?: number, nms?: number): Promise<Detection[]>;
    /**
     * convert darknet image to rgb buffer
     * @param {IMAGE} image
     * @returns {Buffer}
     */
    imageToRGBBuffer(image: Image): Buffer;
    private rgbToDarknet;
    getImageFromDarknetBuffer(buffer: Float32Array, w: number, h: number, c: number): Image;
    /**
     * Transform an RGB buffer to a darknet encoded image
     * @param buffer - rgb buffer
     * @param w - width
     * @param h - height
     * @param c - channels
     * @returns {Promise<IMAGE>}
     */
    RGBBufferToImageAsync(buffer: Buffer, w: number, h: number, c: number): Promise<Image>;
}
export interface IConfig {
    thresh?: number;
    hier_thresh?: number;
    nms?: number;
}
export interface IBufferImage {
    b: Buffer;
    w: number;
    h: number;
    c: number;
}
export declare type IClasses = string[];
export interface IDarknetConfig {
    weights: string;
    config: string;
    names?: string[];
    namefile?: string;
    memory?: number;
}
export interface Detection {
    name: string;
    prob: number;
    box: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
}
export interface Image {
    w: number;
    h: number;
    c: number;
    data: Buffer;
}
