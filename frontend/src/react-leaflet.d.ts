// Type augmentation for react-leaflet v5
import 'react-leaflet';

declare module 'react-leaflet' {
  import { ReactNode, CSSProperties } from 'react';

  export interface MapContainerProps {
    center: [number, number];
    zoom: number;
    style?: CSSProperties;
    zoomControl?: boolean;
    scrollWheelZoom?: boolean;
    children?: ReactNode;
    [key: string]: any;
  }

  export interface TileLayerProps {
    url: string;
    attribution?: string;
    [key: string]: any;
  }

  export interface MarkerProps {
    position: [number, number];
    icon?: any;
    children?: ReactNode;
    eventHandlers?: {
      click?: () => void;
      [key: string]: any;
    };
    [key: string]: any;
  }

  export interface PopupProps {
    children?: ReactNode;
    [key: string]: any;
  }
}

